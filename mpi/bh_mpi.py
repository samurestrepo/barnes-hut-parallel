"""Barnes-Hut con MPI: reparto por Morton y ramas remotas podadas (tarea S4).

  mpiexec -n 4 python mpi/bh_mpi.py --input data/uniform_N20000_s42.txt --steps 20 --out salida/mpi
  mpiexec -n 4 python mpi/bh_mpi.py --particion espacio ...   # reparto ingenuo (demo)

Cada paso de fuerzas:
  1. Allreduce: caja global (mínimos y máximos de todos los rangos).
  2. Cada rango calcula los códigos de Morton de sus partículas.
  3. Gatherv al rango 0, que ordena por código y corta en P trozos; Scatterv.
     (fracción secuencial: los demás esperan)
  4. Cada rango construye el árbol de su trozo.
  5. Allgather de las cajas de todos los rangos.
  6. Poda: cada rango recorre su árbol contra la caja de cada rango remoto. Una
     celda que ya cumple θ vista desde el punto MÁS CERCANO de esa caja la cumple
     para cualquier partícula de la caja: viaja como (cm, masa). Si no, se abre;
     las hojas abiertas viajan como partículas.
  7. Alltoall de cuántos puntos van a cada rango y Alltoallv de los puntos.
  8. Cada rango arma un árbol con sus partículas + lo recibido y calcula las
     aceleraciones de las suyas con el recorrido de py_core (S2).
Al final, Gatherv al rango 0, que devuelve todo al orden de la entrada.
"""
import os

os.environ["NUMBA_NUM_THREADS"] = "1"  # un hilo por rango: el paralelismo lo da MPI
# Perilla (solo Microsoft MPI): cuántas vueltas da un rango esperando en una colectiva
# antes de ceder la CPU. Sin límite, con más rangos que hilos de hardware (8 en 4) los
# que esperan le roban CPU a los que calculan: medido, la comunicación pasó de 8 s a
# ≈ 0.3 s. Debe fijarse ANTES de importar mpi4py (ahí arranca MPI). OpenMPI la ignora.
os.environ.setdefault("MPICH_PROGRESS_SPIN_LIMIT", "16")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from numba import njit  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "py_core"))
import bh_core as bh  # noqa: E402  (también pone common/ en sys.path)
from directo import FASES, escribir_salidas  # noqa: E402
from fisica import leer_particulas  # noqa: E402

# ----------------------------------------------------------------- perillas de la demo
PARTICION = "cantidad"  # --particion: cantidad = N/P partículas por rango (Morton);
                        # espacio = el espacio de códigos en P partes iguales (ingenuo)

# ----------------------------------------------------------------- constantes
COLUMNAS = 8            # cada partícula viaja como x y z vx vy vz m índice_original
FLOATS_POR_PUNTO = 4    # lo que viaja de las ramas remotas: x y z m
BYTES_POR_PUNTO = FLOATS_POR_PUNTO * 8
MAX_CODIGO = 1 << (3 * bh.BITS_MORTON)  # los códigos van de 0 a 2^63 - 1

comm = MPI.COMM_WORLD
RANGO, P = comm.Get_rank(), comm.Get_size()


# ----------------------------------------------------------------- poda (Numba)

@njit(cache=True)
def distancia_punto_caja(px, py, pz, cmin, cmax):
    """Distancia del punto p al punto más cercano de la caja (0 si p está dentro)."""
    d2 = 0.0
    for eje, x in enumerate((px, py, pz)):
        if x < cmin[eje]:
            d2 += (cmin[eje] - x) ** 2
        elif x > cmax[eje]:
            d2 += (x - cmax[eje]) ** 2
    return np.sqrt(d2)


@njit(cache=True)
def celda_toca_caja(arbol, nodo, cmin, cmax):
    """¿La caja de la celda y la caja remota se tocan? Si sí, alguna partícula remota
    podría estar dentro de la celda y el contrato dice que esa celda se abre siempre."""
    for eje in range(3):
        if arbol.centro[nodo, eje] + arbol.mitad[nodo] < cmin[eje]:
            return False
        if arbol.centro[nodo, eje] - arbol.mitad[nodo] > cmax[eje]:
            return False
    return True


@njit(cache=True)
def acepta_para_caja(arbol, nodo, cmin, cmax, theta):
    """Criterio θ con la distancia MÍNIMA de la caja al centro de masa: si se cumple
    para el punto más cercano, se cumple para todas las partículas de la caja."""
    if celda_toca_caja(arbol, nodo, cmin, cmax):
        return False
    d = distancia_punto_caja(arbol.cm[nodo, 0], arbol.cm[nodo, 1], arbol.cm[nodo, 2], cmin, cmax)
    return 2.0 * arbol.mitad[nodo] < theta * d


@njit(cache=True)
def puntos_para_caja(arbol, pos, m, cmin, cmax, theta):
    """Ramas podadas para un rango remoto: filas (x, y, z, masa) a enviarle."""
    salida = np.empty((arbol.ini.shape[0] + pos.shape[0], FLOATS_POR_PUNTO))
    k = 0
    pila = np.empty(bh.TAM_PILA, np.int64)
    pila[0] = 0
    tope = 1
    while tope > 0:
        tope -= 1
        nodo = pila[tope]
        if arbol.masa[nodo] == 0.0:  # rango sin partículas: nada que mandar
            continue
        if acepta_para_caja(arbol, nodo, cmin, cmax, theta):  # lejos: la celda resumida
            salida[k, 0:3] = arbol.cm[nodo]
            salida[k, 3] = arbol.masa[nodo]
            k += 1
        elif arbol.num_hijos[nodo] == 0:  # hoja cercana: sus partículas tal cual
            for j in range(arbol.ini[nodo], arbol.fin[nodo]):
                salida[k, 0:3] = pos[j]
                salida[k, 3] = m[j]
                k += 1
        else:
            for h in range(arbol.primer_hijo[nodo], arbol.primer_hijo[nodo] + arbol.num_hijos[nodo]):
                pila[tope] = h
                tope += 1
    return salida[:k]


# ----------------------------------------------------------------- reparto (rango 0 ordena)

def caja_global(pos, tiempos):
    """Allreduce de mínimos y máximos: todos usan la misma caja para Morton."""
    minimo = pos.min(axis=0) if len(pos) else np.full(3, np.inf)
    maximo = pos.max(axis=0) if len(pos) else np.full(3, -np.inf)
    min_global, max_global = np.empty(3), np.empty(3)
    inicio = time.perf_counter()
    # SINCRONIZACIÓN: colectiva 1 del paso (todos los rangos, mismo orden).
    comm.Allreduce(minimo, min_global, op=MPI.MIN)
    # SINCRONIZACIÓN: colectiva 2 del paso.
    comm.Allreduce(maximo, max_global, op=MPI.MAX)
    tiempos["comunicacion"] += time.perf_counter() - inicio
    return bh.cubo_desde_extremos(min_global, max_global)


def cortes_por_rango(codigos_ordenados, particion):
    """Cuántas partículas le tocan a cada rango (solo en el rango 0)."""
    n = len(codigos_ordenados)
    if particion == "cantidad":  # N/P cada uno: parejo en partículas, aunque estén apiñadas
        cortes = np.linspace(0, n, P + 1).astype(np.int64)
    else:  # espacio: P tramos iguales del espacio de códigos; con corner, casi todo a uno
        limites = np.array([r * (MAX_CODIGO // P) for r in range(1, P)], dtype=np.int64)
        cortes = np.concatenate([[0], np.searchsorted(codigos_ordenados, limites), [n]])
    return np.diff(cortes).astype(np.int64)


def juntar_en_cero(locales, codigos, tiempos):
    """Gatherv de partículas y códigos al rango 0. Devuelve (todas, códigos) en el 0."""
    cuentas = np.zeros(P, np.int64)
    inicio = time.perf_counter()
    # SINCRONIZACIÓN: colectiva 3 del paso: cuántas partículas tiene cada uno.
    comm.Gather(np.array([len(locales)], np.int64), cuentas, root=0)
    todas = np.empty((cuentas.sum(), COLUMNAS)) if RANGO == 0 else None
    todos_codigos = np.empty(cuentas.sum(), np.int64) if RANGO == 0 else None
    # SINCRONIZACIÓN: colectivas 4 y 5: partículas y códigos al rango 0.
    comm.Gatherv(locales, [todas, cuentas * COLUMNAS] if RANGO == 0 else None, root=0)
    comm.Gatherv(codigos, [todos_codigos, cuentas] if RANGO == 0 else None, root=0)
    tiempos["comunicacion"] += time.perf_counter() - inicio
    return todas, todos_codigos


def repartir_desde_cero(todas, todos_codigos, particion, tiempos):
    """El rango 0 ordena por Morton y reparte con Scatterv. Devuelve el trozo propio."""
    cuentas = np.zeros(P, np.int64)
    if RANGO == 0:
        inicio = time.perf_counter()
        orden = bh.ordenar_por_morton(todos_codigos)  # la parte secuencial del diseño
        todas, todos_codigos = todas[orden], todos_codigos[orden]
        cuentas = cortes_por_rango(todos_codigos, particion)
        tiempos["ordenamiento"] += time.perf_counter() - inicio
    mia = np.zeros(1, np.int64)
    inicio = time.perf_counter()
    # SINCRONIZACIÓN: colectivas 6, 7 y 8: los demás rangos esperan aquí mientras el 0 ordena.
    comm.Scatter(cuentas, mia, root=0)
    locales = np.empty((mia[0], COLUMNAS))
    codigos = np.empty(mia[0], np.int64)
    comm.Scatterv([todas, cuentas * COLUMNAS] if RANGO == 0 else None, locales, root=0)  # PARALELO: cada rango recibe su trozo Morton
    comm.Scatterv([todos_codigos, cuentas] if RANGO == 0 else None, codigos, root=0)
    tiempos["comunicacion"] += time.perf_counter() - inicio
    return locales, codigos


def redistribuir(locales, particion, tiempos):
    """Pasos 1–3: caja global, Morton, el 0 ordena y reparte. Devuelve (locales, códigos, caja)."""
    minimo, lado = caja_global(locales[:, 0:3], tiempos)
    inicio = time.perf_counter()
    codigos = bh.codigos_morton(np.ascontiguousarray(locales[:, 0:3]), minimo, lado)
    tiempos["ordenamiento"] += time.perf_counter() - inicio
    todas, todos_codigos = juntar_en_cero(locales, codigos, tiempos)
    locales, codigos = repartir_desde_cero(todas, todos_codigos, particion, tiempos)
    return locales, codigos, (minimo, lado)


# ----------------------------------------------------------------- ramas remotas

def cajas_de_todos(pos, n_nodos, tiempos):
    """Allgather de (mín xyz, máx xyz, partículas, nodos) de cada rango."""
    fila = np.empty(8)
    fila[0:3] = pos.min(axis=0) if len(pos) else np.inf
    fila[3:6] = pos.max(axis=0) if len(pos) else -np.inf
    fila[6], fila[7] = len(pos), n_nodos
    todas = np.empty((P, 8))
    inicio = time.perf_counter()
    # SINCRONIZACIÓN: colectiva 9 del paso.
    comm.Allgather(fila, todas)
    tiempos["comunicacion"] += time.perf_counter() - inicio
    return todas


def armar_envios(arbol, pos, m, cajas, theta):
    """Puntos para cada rango remoto, pegados en un solo búfer, y cuántos floats a cada uno."""
    trozos, cuentas = [], np.zeros(P, np.int64)
    for r in range(P):
        if r == RANGO or cajas[r, 6] == 0:  # a uno mismo o a un rango vacío no se manda nada
            continue
        puntos = puntos_para_caja(arbol, pos, m, cajas[r, 0:3].copy(), cajas[r, 3:6].copy(), theta)  # PARALELO: cada rango poda su árbol
        trozos.append(puntos.ravel())
        cuentas[r] = puntos.size
    envio = np.concatenate(trozos) if trozos else np.empty(0)
    return envio, cuentas


def intercambiar(envio, cuentas_envio, tiempos):
    """Alltoall de las cuentas y Alltoallv de los puntos. Devuelve los puntos recibidos (k, 4)."""
    cuentas_recibo = np.zeros(P, np.int64)
    inicio = time.perf_counter()
    # SINCRONIZACIÓN: colectiva 10: cada rango aprende cuánto le va a llegar de cada otro.
    comm.Alltoall(cuentas_envio, cuentas_recibo)
    recibido = np.empty(cuentas_recibo.sum())
    # SINCRONIZACIÓN: colectiva 11: las ramas podadas. Con colectivas no hay Send/Send
    # cruzados que se bloqueen entre sí: MPI organiza todo el intercambio.
    comm.Alltoallv([envio, cuentas_envio], [recibido, cuentas_recibo])
    tiempos["comunicacion"] += time.perf_counter() - inicio
    return recibido.reshape(-1, FLOATS_POR_PUNTO)


def bytes_arbol_completo(cajas):
    """Lo que habría llegado si cada rango mandara su árbol entero (nodos + partículas)."""
    otros = [r for r in range(P) if r != RANGO]
    return int(sum(cajas[r, 6] + cajas[r, 7] for r in otros) * BYTES_POR_PUNTO)


# ----------------------------------------------------------------- fuerzas de un paso

def fuerzas_locales(locales, codigos, caja, a, tiempos, cuenta):
    """Pasos 4–8: árbol local, poda, intercambio y aceleraciones de las partículas propias."""
    pos = np.ascontiguousarray(locales[:, 0:3])
    m = np.ascontiguousarray(locales[:, 6])
    inicio = time.perf_counter()
    arbol = bh.construir_arbol(codigos, *caja)  # ya vienen ordenados por Morton
    bh.calcular_masas(arbol, pos, m)
    tiempos["construccion"] += time.perf_counter() - inicio
    cajas = cajas_de_todos(pos, len(arbol.ini), tiempos)
    inicio = time.perf_counter()
    envio, cuentas_envio = armar_envios(arbol, pos, m, cajas, a.theta)
    tiempos["construccion"] += time.perf_counter() - inicio
    recibido = intercambiar(envio, cuentas_envio, tiempos)
    cuenta["bytes_recibidos"] += recibido.nbytes
    cuenta["bytes_arbol_completo"] += bytes_arbol_completo(cajas)
    acc, interacciones = aceleraciones_con_recibido(pos, m, recibido, a, tiempos)  # PARALELO: cada rango, solo las suyas
    cuenta["interacciones"] += int(interacciones.sum())
    cuenta["particulas"] += len(pos)
    return acc


def aceleraciones_con_recibido(pos, m, recibido, a, tiempos):
    """Árbol de (propias + recibidas) y recorrido solo para las propias."""
    if len(pos) == 0:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    inicio = time.perf_counter()
    pos_todo = np.ascontiguousarray(np.vstack([pos, recibido[:, 0:3]]))
    m_todo = np.concatenate([m, recibido[:, 3]])
    orden, pos_ord, m_ord, arbol = bh.preparar_arbol(pos_todo, m_todo, dict.fromkeys(FASES, 0.0))
    posicion_ordenada = np.empty(len(orden), np.int64)
    posicion_ordenada[orden] = np.arange(len(orden))
    objetivos = posicion_ordenada[:len(pos)]  # dónde quedó cada partícula propia
    medio = time.perf_counter()
    acc, interacciones = bh.aceleraciones_lista(objetivos, arbol, pos_ord, m_ord, a.theta, a.eps)
    tiempos["construccion"] += medio - inicio
    tiempos["recorrido"] += time.perf_counter() - medio
    return acc, interacciones


# ----------------------------------------------------------------- juntar resultados

def juntar_en_orden_original(filas, indices, n_total):
    """Gatherv al rango 0 y vuelta al orden de la entrada (si no, el verificador falla)."""
    cuentas = np.zeros(P, np.int64)
    columnas = filas.shape[1]
    # SINCRONIZACIÓN: colectivas de cierre, las mismas en todos los rangos.
    comm.Gather(np.array([len(filas)], np.int64), cuentas, root=0)
    todas = np.empty((n_total, columnas)) if RANGO == 0 else None
    todos_idx = np.empty(n_total, np.int64) if RANGO == 0 else None
    comm.Gatherv(np.ascontiguousarray(filas), [todas, cuentas * columnas] if RANGO == 0 else None, root=0)
    comm.Gatherv(np.ascontiguousarray(indices), [todos_idx, cuentas] if RANGO == 0 else None, root=0)
    if RANGO != 0:
        return None
    salida = np.empty_like(todas)
    salida[todos_idx] = todas
    return salida


# ----------------------------------------------------------------- simulación

def simular(locales, n_total, a, tiempos, cuenta):
    """Leapfrog KDK. Devuelve (acc0 en el 0, partículas locales al final)."""
    locales, codigos, caja = redistribuir(locales, a.particion, tiempos)
    acc = fuerzas_locales(locales, codigos, caja, a, tiempos, cuenta)
    acc0 = juntar_en_orden_original(acc, locales[:, 7].astype(np.int64), n_total)
    for _ in range(a.steps):
        inicio = time.perf_counter()
        vel = locales[:, 3:6]  # vista: kick y drift cambian `locales` en su sitio
        bh.medio_kick(vel, acc, a.dt)
        locales[:, 0:3] += a.dt * vel
        tiempos["integracion"] += time.perf_counter() - inicio
        # Las partículas se movieron: nuevo reparto (con su velocidad) y nuevo árbol.
        locales, codigos, caja = redistribuir(locales, a.particion, tiempos)
        acc = fuerzas_locales(locales, codigos, caja, a, tiempos, cuenta)
        inicio = time.perf_counter()
        bh.medio_kick(locales[:, 3:6], acc, a.dt)
        tiempos["integracion"] += time.perf_counter() - inicio
    return acc0, locales


def calentar(a):
    """Compila Numba sin medir. El 0 compila y escribe la caché; los demás la cargan después,
    para que dos procesos no escriban el mismo archivo de caché a la vez."""
    pos = np.random.default_rng(0).random((64, 3))
    m = np.full(64, 1.0 / 64)
    if RANGO != 0:
        comm.Barrier()  # SINCRONIZACIÓN: espera a que el 0 termine de compilar
    bh.calentar_numba(pos, m, a.theta, a.eps)
    arbol = bh.construir_arbol(np.sort(bh.codigos_morton(pos, np.zeros(3), 1.0)), np.zeros(3), 1.0)
    bh.calcular_masas(arbol, pos, m)
    puntos_para_caja(arbol, pos, m, np.full(3, 2.0), np.full(3, 3.0), a.theta)
    if RANGO == 0:
        comm.Barrier()  # SINCRONIZACIÓN: misma barrera, ahora del lado del 0
    # SINCRONIZACIÓN: todos salen juntos del calentamiento. Sin esta barrera, el 0 mediría
    # como "comunicación" la espera mientras los demás cargan la caché de Numba.
    comm.Barrier()


# ----------------------------------------------------------------- programa

def leer_argumentos():
    p = argparse.ArgumentParser(description="Barnes-Hut con MPI (S4)")
    p.add_argument("--input", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--dt", type=float, default=bh.DT)
    p.add_argument("--theta", type=float, default=bh.THETA)
    p.add_argument("--eps", type=float, default=bh.EPS)
    p.add_argument("--workers", type=int, default=1)  # contrato: se ignora, lo da mpiexec -n
    p.add_argument("--particion", choices=["cantidad", "espacio"], default=PARTICION)
    p.add_argument("--out", required=True)
    return p.parse_args()


def leer_en_cero(ruta, tiempos):
    """Solo el rango 0 lee; los demás empiezan vacíos y reciben su trozo en el primer reparto."""
    if RANGO != 0:
        return np.empty((0, COLUMNAS)), 0
    inicio = time.perf_counter()
    pos, vel, m = leer_particulas(ruta)
    locales = np.column_stack([pos, vel, m, np.arange(len(pos), dtype=np.float64)])
    tiempos["lectura"] = time.perf_counter() - inicio
    return locales, len(pos)


def juntar_metricas(tiempos, cuenta):
    """Gather de fases y cuentas de cada rango (filas en el 0)."""
    fila = np.array([tiempos[f] for f in FASES[:-1]] + [cuenta["interacciones"], cuenta["particulas"],
                                                        cuenta["bytes_recibidos"], cuenta["bytes_arbol_completo"]])
    todas = np.empty((P, len(fila))) if RANGO == 0 else None
    comm.Gather(fila, todas, root=0)  # SINCRONIZACIÓN: última colectiva
    return todas


def tiempos_del_mas_lento(todas):
    """Cada fase, la del rango que más tardó en ella: es la que fija el reloj de pared."""
    return {fase: float(todas[:, i].max()) for i, fase in enumerate(FASES[:-1])}


def escribir_metricas(prefijo, todas, pasos):
    """Agrega los bytes a times.csv y escribe workers.csv (una fila por rango)."""
    nf = len(FASES) - 1
    with open(f"{prefijo}_times.csv", "a", encoding="utf-8") as f:
        f.write(f"bytes_recibidos,{int(todas[:, nf + 2].sum())}\n")
        f.write(f"bytes_arbol_completo,{int(todas[:, nf + 3].sum())}\n")
    with open(f"{prefijo}_workers.csv", "w", encoding="utf-8") as f:
        f.write("trabajador,segundos,interacciones,particulas_por_paso,bytes_recibidos,bytes_arbol_completo\n")
        for r in range(P):
            fila = todas[r]
            f.write(f"{r},{fila[FASES.index('recorrido')]:.6f},{int(fila[nf])},"
                    f"{fila[nf + 1] / (pasos + 1):.1f},{int(fila[nf + 2])},{int(fila[nf + 3])}\n")


def main():
    a = leer_argumentos()
    tiempos = dict.fromkeys(FASES, 0.0)
    cuenta = dict.fromkeys(["interacciones", "particulas", "bytes_recibidos", "bytes_arbol_completo"], 0)
    inicio = time.perf_counter()
    locales, n_total = leer_en_cero(a.input, tiempos)
    lectura = time.perf_counter() - inicio
    calentar(a)
    inicio = time.perf_counter()
    acc0, locales = simular(locales, n_total, a, tiempos, cuenta)
    final = juntar_en_orden_original(locales[:, 0:7], locales[:, 7].astype(np.int64), n_total)
    simulacion = time.perf_counter() - inicio
    todas = juntar_metricas(tiempos, cuenta)
    if RANGO == 0:
        tiempos_salida = tiempos_del_mas_lento(todas)
        tiempos_salida["total"] = lectura + simulacion  # reloj de pared del rango 0
        escribir_salidas(a.out, acc0, final[:, 0:3], final[:, 3:6], final[:, 6], tiempos_salida)
        escribir_metricas(a.out, todas, a.steps)

if __name__ == "__main__":
    main()
