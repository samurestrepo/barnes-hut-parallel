"""Barnes-Hut secuencial en Python + Numba (tarea S2).

Es también el núcleo de py_mp (S3) y mpi (S4): cada fase es una función con
entradas y salidas en arreglos de NumPy, sin estado global, para que esas
versiones la importen y repartan el trabajo sin copiar código.

  python py_core/bh_core.py --input data/uniform_N2000_s42.txt --steps 10 --out salida/bh
  NUMBA_DISABLE_JIT=1 python py_core/bh_core.py ...   # depurar sin compilar (N pequeño)

Fases de un cálculo de fuerzas:
  caja_raiz → codigos_morton → ordenar_por_morton → construir_arbol
  → calcular_masas → aceleraciones_rango → (desordenar) → kick / drift
"""
import argparse
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from numba import njit

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "common"))
from directo import FASES, escribir_salidas  # noqa: E402  (mismo formato %.17g que el verificador)
from fisica import leer_particulas  # noqa: E402

# ----------------------------------------------------------------- perillas de la demo
THETA = 0.5       # criterio de apertura por defecto (--theta). θ = 0 → suma directa exacta
EPS = 0.01        # suavizado por defecto (--eps)
DT = 0.001        # paso por defecto (--dt); con corner/clusters usar 2.5e-4 (contrato)
HOJA_MAX = 8      # contrato: hasta 8 partículas por hoja; más grande = árbol más bajo
G = 1.0

# ----------------------------------------------------------------- constantes internas
BITS_MORTON = 21  # bits por eje: 3 × 21 = 63 bits, caben en un int64 con signo
# La pila de recorrido y de construcción guarda a lo sumo 7 hermanos pendientes por
# nivel más los 8 hijos del último: 7 × 21 + 8 = 155 < 256. Nunca se desborda.
TAM_PILA = 256
# Tolerancia relativa al decidir si una partícula está dentro de la caja de una celda:
# el centro de la celda se obtiene partiendo a la mitad 21 veces y acumula redondeo.
TOL_CAJA = 1e-9

# Árbol en arreglos planos: el nodo k está descrito por la posición k de cada arreglo.
# Sus partículas son las ordenadas [ini[k], fin[k]) y sus hijos, si tiene, son los
# nodos contiguos primer_hijo[k] … primer_hijo[k] + num_hijos[k] - 1.
Arbol = namedtuple("Arbol", ["ini", "fin", "primer_hijo", "num_hijos", "nivel",
                             "centro", "mitad", "masa", "cm"])


# ----------------------------------------------------------------- caja y Morton

def cubo_desde_extremos(minimo, maximo):
    """Cubo (esquina mínima, lado) que contiene la caja [minimo, maximo].

    MPI la usa con los extremos globales que da allreduce; aquí, con los locales."""
    lado = float(np.max(maximo - minimo))
    if lado == 0.0:  # todas en el mismo punto: cualquier lado sirve
        lado = 1.0
    return np.ascontiguousarray(minimo, dtype=np.float64), lado


def caja_raiz(pos):
    """Caja raíz recalculada en cada paso: nunca se supone el cubo unitario."""
    return cubo_desde_extremos(pos.min(axis=0), pos.max(axis=0))


@njit(cache=True)
def cuantizar(x, minimo, lado):
    """Coordenada → entero de 21 bits (celda de la malla de 2^21 por eje)."""
    celdas = 1 << BITS_MORTON
    q = int((x - minimo) / lado * celdas)
    # La partícula del borde superior caería en la celda 2^21, que no existe.
    return min(max(q, 0), celdas - 1)


@njit(cache=True)
def intercalar_bits(qx, qy, qz):
    """Código de Morton: bits x y z intercalados (x el más alto de cada trío).

    Con un bucle en vez de los "números mágicos" para que se lea qué hace."""
    codigo = 0
    for b in range(BITS_MORTON):
        codigo |= ((qx >> b) & 1) << (3 * b + 2)
        codigo |= ((qy >> b) & 1) << (3 * b + 1)
        codigo |= ((qz >> b) & 1) << (3 * b)
    return codigo


@njit(cache=True)
def codigos_morton(pos, minimo, lado):
    """Un código por partícula. Partículas cercanas en el espacio → códigos cercanos."""
    n = pos.shape[0]
    codigos = np.empty(n, np.int64)
    for i in range(n):
        qx = cuantizar(pos[i, 0], minimo[0], lado)
        qy = cuantizar(pos[i, 1], minimo[1], lado)
        qz = cuantizar(pos[i, 2], minimo[2], lado)
        codigos[i] = intercalar_bits(qx, qy, qz)
    return codigos


def ordenar_por_morton(codigos):
    """Índices que ordenan por código. `orden[k]` = índice original de la k-ésima.

    Estable: con códigos repetidos se conserva el orden de la entrada (reproducible)."""
    return np.argsort(codigos, kind="stable")


# ----------------------------------------------------------------- construcción del árbol

def arbol_vacio(capacidad):
    """Reserva los arreglos de un árbol con `capacidad` nodos."""
    return Arbol(ini=np.zeros(capacidad, np.int64), fin=np.zeros(capacidad, np.int64),
                 primer_hijo=np.zeros(capacidad, np.int64),
                 num_hijos=np.zeros(capacidad, np.int64), nivel=np.zeros(capacidad, np.int64),
                 centro=np.zeros((capacidad, 3)), mitad=np.zeros(capacidad),
                 masa=np.zeros(capacidad), cm=np.zeros((capacidad, 3)))


@njit(cache=True)
def octante(codigo, nivel):
    """Hijo (0–7) al que va un código dentro de una celda del nivel dado."""
    return (codigo >> (3 * (BITS_MORTON - 1 - nivel))) & 7


@njit(cache=True)
def fin_del_octante(codigos, k, fin, nivel):
    """Como los códigos están ordenados, cada hijo es un tramo contiguo: lo recorre."""
    oct_k = octante(codigos[k], nivel)
    while k < fin and octante(codigos[k], nivel) == oct_k:
        k += 1
    return k


@njit(cache=True)
def crear_hijo(arbol, padre, hijo, ini, fin, oct_hijo):
    """Llena el nodo `hijo`: rango de partículas y caja (la mitad de la del padre)."""
    arbol.ini[hijo] = ini
    arbol.fin[hijo] = fin
    arbol.nivel[hijo] = arbol.nivel[padre] + 1
    arbol.num_hijos[hijo] = 0
    mitad = arbol.mitad[padre] / 2.0
    arbol.mitad[hijo] = mitad
    for eje in range(3):  # bit 2 = x, bit 1 = y, bit 0 = z (como en intercalar_bits)
        lado_alto = (oct_hijo >> (2 - eje)) & 1
        arbol.centro[hijo, eje] = arbol.centro[padre, eje] + (mitad if lado_alto else -mitad)


@njit(cache=True)
def dividir_nodo(arbol, nodo, codigos, n_nodos, pila, tope):
    """Crea los hijos no vacíos de `nodo` y los apila. Devuelve (n_nodos, tope) o (-1, tope)."""
    capacidad = arbol.ini.shape[0]
    arbol.primer_hijo[nodo] = n_nodos
    k = arbol.ini[nodo]
    while k < arbol.fin[nodo]:
        fin_hijo = fin_del_octante(codigos, k, arbol.fin[nodo], arbol.nivel[nodo])
        if n_nodos == capacidad:
            return -1, tope  # no cupo: el envoltorio duplica la capacidad y repite
        crear_hijo(arbol, nodo, n_nodos, k, fin_hijo, octante(codigos[k], arbol.nivel[nodo]))
        pila[tope] = n_nodos
        tope += 1
        n_nodos += 1
        k = fin_hijo
    arbol.num_hijos[nodo] = n_nodos - arbol.primer_hijo[nodo]
    return n_nodos, tope


@njit(cache=True)
def llenar_arbol(codigos, minimo, lado, arbol):
    """Octree sobre códigos YA ordenados, con pila explícita. Devuelve nº de nodos o -1."""
    arbol.ini[0] = 0
    arbol.fin[0] = codigos.shape[0]
    arbol.nivel[0] = 0
    arbol.num_hijos[0] = 0
    arbol.mitad[0] = lado / 2.0
    for eje in range(3):
        arbol.centro[0, eje] = minimo[eje] + lado / 2.0
    pila = np.empty(TAM_PILA, np.int64)
    pila[0] = 0
    tope = 1
    n_nodos = 1
    while tope > 0:
        tope -= 1
        nodo = pila[tope]
        cuantas = arbol.fin[nodo] - arbol.ini[nodo]
        # Hoja si caben, o si ya no hay más bits: Morton de 21 bits puede repetir códigos.
        if cuantas <= HOJA_MAX or arbol.nivel[nodo] == BITS_MORTON:
            continue
        n_nodos, tope = dividir_nodo(arbol, nodo, codigos, n_nodos, pila, tope)
        if n_nodos < 0:
            return -1
    return n_nodos


def construir_arbol(codigos_ordenados, minimo, lado):
    """Árbol listo para calcular_masas. Empieza con 2N+64 nodos y duplica si no cabe."""
    capacidad = 2 * len(codigos_ordenados) + 64
    while True:
        arbol = arbol_vacio(capacidad)
        n_nodos = llenar_arbol(codigos_ordenados, minimo, lado, arbol)
        if n_nodos > 0:
            return Arbol(*(campo[:n_nodos] for campo in arbol))
        capacidad *= 2


@njit(cache=True)
def calcular_masas(arbol, pos, m):
    """Masa y centro de masa de cada nodo, de las hojas hacia la raíz.

    Los hijos siempre se crean después que su padre (índice mayor), así que recorrer
    los nodos al revés garantiza que los hijos ya están listos al llegar al padre."""
    for nodo in range(arbol.ini.shape[0] - 1, -1, -1):
        masa, cx, cy, cz = 0.0, 0.0, 0.0, 0.0
        if arbol.num_hijos[nodo] == 0:
            for j in range(arbol.ini[nodo], arbol.fin[nodo]):
                masa += m[j]
                cx += m[j] * pos[j, 0]
                cy += m[j] * pos[j, 1]
                cz += m[j] * pos[j, 2]
        else:
            for h in range(arbol.primer_hijo[nodo], arbol.primer_hijo[nodo] + arbol.num_hijos[nodo]):
                masa += arbol.masa[h]
                cx += arbol.masa[h] * arbol.cm[h, 0]
                cy += arbol.masa[h] * arbol.cm[h, 1]
                cz += arbol.masa[h] * arbol.cm[h, 2]
        arbol.masa[nodo] = masa
        if masa > 0.0:  # S4 puede traer masas 0 en celdas vacías de relleno
            arbol.cm[nodo, 0], arbol.cm[nodo, 1], arbol.cm[nodo, 2] = cx / masa, cy / masa, cz / masa


# ----------------------------------------------------------------- recorrido (fuerzas)

@njit(cache=True)
def termino_gravedad(px, py, pz, qx, qy, qz, mq, eps2):
    """Aceleración sobre p causada por una masa mq en q (fórmula del contrato)."""
    dx, dy, dz = qx - px, qy - py, qz - pz
    r2 = dx * dx + dy * dy + dz * dz + eps2
    f = G * mq / (r2 * np.sqrt(r2))
    return f * dx, f * dy, f * dz


@njit(cache=True)
def acepta_celda(arbol, nodo, px, py, pz, theta):
    """¿La celda cuenta como una sola partícula vista desde p?

    Si p está dentro de la caja de la celda se abre SIEMPRE: con θ ≳ 0.58 el criterio
    podría aceptarla y p se atraería a sí misma (contrato)."""
    mitad = arbol.mitad[nodo]
    margen = mitad * (1.0 + TOL_CAJA)
    if (abs(px - arbol.centro[nodo, 0]) <= margen and abs(py - arbol.centro[nodo, 1]) <= margen
            and abs(pz - arbol.centro[nodo, 2]) <= margen):
        return False
    dx = arbol.cm[nodo, 0] - px
    dy = arbol.cm[nodo, 1] - py
    dz = arbol.cm[nodo, 2] - pz
    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    # lado / d < θ, escrito sin dividir para que d = 0 no dé infinito.
    return 2.0 * mitad < theta * d


@njit(cache=True)
def aceleracion_de(k, arbol, pos, m, theta, eps, pila):
    """Aceleración de la partícula ordenada k recorriendo el árbol con pila explícita.

    Devuelve (ax, ay, az, interacciones). Sin recursión: Numba la compila mejor y
    la pila tiene un tamaño fijo conocido (TAM_PILA)."""
    px, py, pz = pos[k, 0], pos[k, 1], pos[k, 2]
    eps2 = eps * eps
    ax, ay, az = 0.0, 0.0, 0.0
    interacciones = 0
    pila[0] = 0
    tope = 1
    while tope > 0:
        tope -= 1
        nodo = pila[tope]
        if acepta_celda(arbol, nodo, px, py, pz, theta):  # lejos: una sola masa en el cm
            fx, fy, fz = termino_gravedad(px, py, pz, arbol.cm[nodo, 0], arbol.cm[nodo, 1],
                                          arbol.cm[nodo, 2], arbol.masa[nodo], eps2)
            ax, ay, az = ax + fx, ay + fy, az + fz
            interacciones += 1
        elif arbol.num_hijos[nodo] == 0:  # hoja cercana: suma directa con sus partículas
            for j in range(arbol.ini[nodo], arbol.fin[nodo]):
                if j != k:  # una partícula no se atrae a sí misma
                    fx, fy, fz = termino_gravedad(px, py, pz, pos[j, 0], pos[j, 1], pos[j, 2], m[j], eps2)
                    ax, ay, az = ax + fx, ay + fy, az + fz
                    interacciones += 1
        else:  # celda cercana: se abre
            for h in range(arbol.primer_hijo[nodo], arbol.primer_hijo[nodo] + arbol.num_hijos[nodo]):
                pila[tope] = h
                tope += 1
    return ax, ay, az, interacciones


@njit(cache=True)
def aceleraciones_lista(objetivos, arbol, pos, m, theta, eps):
    """Aceleraciones de las partículas ordenadas `objetivos` (MPI pasa solo las locales)."""
    acc = np.zeros((objetivos.shape[0], 3))
    interacciones = np.zeros(objetivos.shape[0], np.int64)
    pila = np.empty(TAM_PILA, np.int64)
    for t in range(objetivos.shape[0]):
        ax, ay, az, n_int = aceleracion_de(objetivos[t], arbol, pos, m, theta, eps, pila)
        acc[t, 0], acc[t, 1], acc[t, 2] = ax, ay, az
        interacciones[t] = n_int
    return acc, interacciones


def aceleraciones_rango(ini, fin, arbol, pos, m, theta, eps):
    """Aceleraciones de las partículas ordenadas [ini, fin): el trozo de un trabajador."""
    return aceleraciones_lista(np.arange(ini, fin, dtype=np.int64), arbol, pos, m, theta, eps)


# ----------------------------------------------------------------- integración (leapfrog KDK)

def medio_kick(vel, acc, dt):
    """v += a·dt/2. Se hace dos veces por paso: antes y después del drift."""
    vel += 0.5 * dt * acc


def drift(pos, vel, dt):
    """x += v·dt: paso completo de posición con la velocidad de medio paso."""
    pos += dt * vel


# ----------------------------------------------------------------- un cálculo completo de fuerzas

def preparar_arbol(pos, m, tiempos):
    """Caja → Morton → orden → árbol con masas. Devuelve (orden, pos_ord, m_ord, arbol)."""
    inicio = time.perf_counter()
    minimo, lado = caja_raiz(pos)
    codigos = codigos_morton(pos, minimo, lado)
    orden = ordenar_por_morton(codigos)
    # Copias contiguas en orden Morton: el recorrido lee vecinos en memoria.
    pos_ord = np.ascontiguousarray(pos[orden])
    m_ord = np.ascontiguousarray(m[orden])
    codigos_ord = codigos[orden]
    medio = time.perf_counter()
    arbol = construir_arbol(codigos_ord, minimo, lado)
    calcular_masas(arbol, pos_ord, m_ord)
    tiempos["ordenamiento"] += medio - inicio
    tiempos["construccion"] += time.perf_counter() - medio
    return orden, pos_ord, m_ord, arbol


def fuerzas(pos, m, theta, eps, tiempos):
    """Aceleraciones en el ORDEN DE LA ENTRADA (y las interacciones por partícula)."""
    orden, pos_ord, m_ord, arbol = preparar_arbol(pos, m, tiempos)
    inicio = time.perf_counter()
    acc_ord, inter_ord = aceleraciones_rango(0, len(pos), arbol, pos_ord, m_ord, theta, eps)
    acc = np.empty_like(acc_ord)
    acc[orden] = acc_ord  # devolver al orden de la entrada: si no, el verificador falla
    inter = np.empty_like(inter_ord)
    inter[orden] = inter_ord
    tiempos["recorrido"] += time.perf_counter() - inicio
    return acc, inter


def calentar_numba(pos, m, theta, eps):
    """Paso de calentamiento NO medido: compila (o carga de la caché) todas las funciones.

    Con pocas partículas basta: Numba compila por tipos, no por tamaño."""
    n = min(len(pos), 64)
    fuerzas(np.ascontiguousarray(pos[:n]), np.ascontiguousarray(m[:n]), theta, eps,
            dict.fromkeys(FASES, 0.0))


# ----------------------------------------------------------------- programa

def leer_argumentos():
    p = argparse.ArgumentParser(description="Barnes-Hut secuencial (S2)")
    p.add_argument("--input", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--theta", type=float, default=THETA)
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--workers", type=int, default=1)  # contrato: se acepta; aquí es secuencial
    p.add_argument("--out", required=True)
    return p.parse_args()


def simular(pos, vel, m, a, tiempos):
    """Leapfrog kick-drift-kick. Devuelve las aceleraciones del paso 0 (acc0.txt)."""
    acc, _ = fuerzas(pos, m, a.theta, a.eps, tiempos)
    acc0 = acc.copy()
    for _ in range(a.steps):
        inicio = time.perf_counter()
        medio_kick(vel, acc, a.dt)   # medio kick con las aceleraciones viejas
        drift(pos, vel, a.dt)        # paso completo de posición
        tiempos["integracion"] += time.perf_counter() - inicio
        acc, _ = fuerzas(pos, m, a.theta, a.eps, tiempos)  # árbol nuevo: las partículas se movieron
        inicio = time.perf_counter()
        medio_kick(vel, acc, a.dt)   # medio kick con las nuevas: sin él la energía deriva
        tiempos["integracion"] += time.perf_counter() - inicio
    return acc0


def main():
    a = leer_argumentos()
    tiempos = dict.fromkeys(FASES, 0.0)  # comunicacion queda en 0: no hay
    inicio = time.perf_counter()
    pos, vel, m = leer_particulas(a.input)
    tiempos["lectura"] = time.perf_counter() - inicio
    calentar_numba(pos, m, a.theta, a.eps)
    inicio_medido = time.perf_counter()
    acc0 = simular(pos, vel, m, a, tiempos)
    # total = lectura + simulación; el calentamiento queda fuera a propósito.
    tiempos["total"] = tiempos["lectura"] + time.perf_counter() - inicio_medido
    escribir_salidas(a.out, acc0, pos, vel, m, tiempos)


if __name__ == "__main__":
    main()
