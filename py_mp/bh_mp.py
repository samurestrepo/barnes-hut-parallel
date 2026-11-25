"""Barnes-Hut con multiprocessing: árbol compartido vs árbol local (tarea S3).

  python py_mp/bh_mp.py --tree shared --workers 4 --input data/uniform_N20000_s42.txt --steps 20 --out salida/mp
  python py_mp/bh_mp.py --tree local  --workers 4 ...

Dos formas de darles el árbol a los trabajadores:
  shared: el principal construye el árbol y lo copia en memoria compartida
          (multiprocessing.shared_memory); los trabajadores solo lo leen.
  local:  cada trabajador reconstruye el mismo árbol a partir de las posiciones
          compartidas. Trabajo repetido, pero no hay que copiar el árbol.

En los dos modos las posiciones, las masas y las aceleraciones viven en memoria
compartida en el ORDEN DE LA ENTRADA; el principal integra sobre ellas en su
sitio. Las fuerzas se parten en TROZOS_POR_PROCESO × workers trozos contiguos
(en orden Morton) que se reparten con imap_unordered: el trabajador que se
libera toma el siguiente trozo, así un trozo caro (un cúmulo) no frena a todos.
El árbol y las fórmulas son los de py_core/bh_core.py (S2).
"""
import os

# Un hilo por proceso: el paralelismo lo da multiprocessing, no Numba. Debe ir
# ANTES de importar numba (lo importa bh_core), también en los trabajadores.
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool, shared_memory  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "py_core"))
import bh_core as bh  # noqa: E402  (también pone common/ en sys.path)
from directo import FASES, escribir_salidas  # noqa: E402
from fisica import leer_particulas  # noqa: E402

# ----------------------------------------------------------------- perillas de la demo
TROZOS_POR_PROCESO = 6   # 4–8: más trozos = mejor balanceo con corner, más tareas que enviar
MODO_ARBOL = "shared"    # por defecto (--tree shared|local)

# Estado de cada trabajador (un proceso aparte, así que no se comparte con nadie).
_TRABAJADOR = {}


# ----------------------------------------------------------------- memoria compartida

def crear_bloque(forma, dtype):
    """Crea un bloque de memoria compartida y un arreglo de NumPy encima."""
    tam = max(1, int(np.prod(forma)) * np.dtype(dtype).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=tam)
    return shm, np.ndarray(forma, dtype=dtype, buffer=shm.buf)


def adjuntar_bloque(nombre, forma, dtype):
    """Abre desde un trabajador un bloque que creó el principal (sin copiarlo)."""
    try:  # Python ≥ 3.13: que el trabajador no lo borre al salir; el dueño es el principal
        shm = shared_memory.SharedMemory(name=nombre, track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=nombre)
    return shm, np.ndarray(forma, dtype=dtype, buffer=shm.buf)


def describir(bloques):
    """{campo: (nombre, forma, dtype)}: lo único que viaja para que el otro lo abra."""
    return {campo: (shm.name, arr.shape, arr.dtype.str) for campo, (shm, arr) in bloques.items()}


def liberar(bloques):
    """close + unlink de cada bloque. Se llama en un finally: también si hubo error."""
    for shm, _ in bloques.values():
        shm.close()
        shm.unlink()
    bloques.clear()


def bloques_particulas(n, modo):
    """Bloques fijos: posiciones, masas, aceleraciones; en shared, también el orden Morton."""
    bloques = {"pos": crear_bloque((n, 3), np.float64), "m": crear_bloque((n,), np.float64),
               "acc": crear_bloque((n, 3), np.float64)}
    if modo == "shared":
        bloques["orden"] = crear_bloque((n,), np.int64)
        bloques["pos_ord"] = crear_bloque((n, 3), np.float64)
        bloques["m_ord"] = crear_bloque((n,), np.float64)
    return bloques


def bloques_arbol(capacidad):
    """Un bloque por arreglo del árbol, con espacio para `capacidad` nodos."""
    plantilla = bh.arbol_vacio(1)
    return {campo: crear_bloque((capacidad,) + arr.shape[1:], arr.dtype)
            for campo, arr in zip(bh.Arbol._fields, plantilla)}


# ----------------------------------------------------------------- lado del principal

def publicar_arbol(comp, pos, m, tiempos):
    """Modo shared: el principal construye el árbol y lo copia a memoria compartida.

    Devuelve (descripción del árbol, nº de nodos). Si el árbol no cabe en los bloques,
    los recrea más grandes; los trabajadores lo notan porque cambian los nombres."""
    orden, pos_ord, m_ord, arbol = bh.preparar_arbol(pos, m, tiempos)
    inicio = time.perf_counter()
    n_nodos = len(arbol.ini)
    if n_nodos > comp["capacidad"]:
        liberar(comp["arbol"])
        comp["capacidad"] = 2 * n_nodos
        comp["arbol"] = bloques_arbol(comp["capacidad"])
    for campo, arr in zip(bh.Arbol._fields, arbol):
        comp["arbol"][campo][1][:n_nodos] = arr
    comp["part"]["orden"][1][:] = orden
    comp["part"]["pos_ord"][1][:] = pos_ord
    comp["part"]["m_ord"][1][:] = m_ord
    # Copiar el árbol a la memoria compartida es el costo de "compartir": va como comunicación.
    tiempos["comunicacion"] += time.perf_counter() - inicio
    return describir(comp["arbol"]), n_nodos


def repartir_trozos(n, n_trozos):
    """Rangos [ini, fin) contiguos en orden Morton: cada trozo es una región compacta."""
    cortes = np.linspace(0, n, min(n, n_trozos) + 1).astype(np.int64)
    return [(int(cortes[t]), int(cortes[t + 1])) for t in range(len(cortes) - 1)]


def fuerzas_paralelas(pool, comp, paso, a, tiempos, registro):
    """Calcula comp acc (orden de la entrada) repartiendo los trozos entre el Pool."""
    pos, m = comp["part"]["pos"][1], comp["part"]["m"][1]
    descr_arbol, n_nodos = None, 0
    if a.tree == "shared":
        descr_arbol, n_nodos = publicar_arbol(comp, pos, m, tiempos)
    tareas = [(ini, fin, paso, descr_arbol, n_nodos, a.theta, a.eps)
              for ini, fin in repartir_trozos(len(pos), a.workers * TROZOS_POR_PROCESO)]
    inicio = time.perf_counter()
    construccion_trabajadores = {}
    # PARALELO: cada trozo va al primer trabajador libre (balanceo dinámico).
    for pid, seg_arbol, seg_fuerzas, interacciones in pool.imap_unordered(tarea_trozo, tareas):
        anotar(registro, pid, seg_arbol + seg_fuerzas, interacciones)
        construccion_trabajadores[pid] = construccion_trabajadores.get(pid, 0.0) + seg_arbol
    # SINCRONIZACIÓN: el for termina cuando llegó el resultado de TODOS los trozos. Es la
    # barrera que garantiza que acc está completa antes de integrar y de mover pos.
    pared = time.perf_counter() - inicio
    # SUPOSICIÓN (modo local): la construcción repetida corre en paralelo en cada trabajador;
    # se cuenta su promedio como "construccion" y el resto de la fase como "recorrido".
    seg_construccion = np.mean(list(construccion_trabajadores.values())) if a.tree == "local" else 0.0
    tiempos["construccion"] += seg_construccion
    tiempos["recorrido"] += pared - seg_construccion


def anotar(registro, pid, segundos, interacciones):
    """Suma por trabajador; el número de trabajador es el orden en que apareció su pid."""
    fila = registro.setdefault(pid, {"trabajador": len(registro), "segundos": 0.0,
                                     "interacciones": 0, "trozos": 0})
    fila["segundos"] += segundos
    fila["interacciones"] += interacciones
    fila["trozos"] += 1


# ----------------------------------------------------------------- lado del trabajador

def iniciar_trabajador(modo, descr_particulas):
    """Se corre una vez al crear cada proceso del Pool: abre la memoria compartida y
    carga Numba de la caché, para que eso no caiga dentro de la medición."""
    _TRABAJADOR["modo"] = modo
    _TRABAJADOR["part"] = {c: adjuntar_bloque(*d) for c, d in descr_particulas.items()}
    _TRABAJADOR["arbol"], _TRABAJADOR["nombres_arbol"] = {}, None
    _TRABAJADOR["paso_local"] = -1
    pos = _TRABAJADOR["part"]["pos"][1]
    bh.calentar_numba(pos, _TRABAJADOR["part"]["m"][1], bh.THETA, bh.EPS)


def arbol_compartido(descr_arbol, n_nodos):
    """Vistas (sin copia) del árbol que publicó el principal. Reabre si cambió de bloques."""
    nombres = tuple(d[0] for d in descr_arbol.values())
    if nombres != _TRABAJADOR["nombres_arbol"]:
        for shm, _ in _TRABAJADOR["arbol"].values():
            shm.close()
        _TRABAJADOR["arbol"] = {c: adjuntar_bloque(*d) for c, d in descr_arbol.items()}
        _TRABAJADOR["nombres_arbol"] = nombres
    arbol = bh.Arbol(*(_TRABAJADOR["arbol"][c][1][:n_nodos] for c in bh.Arbol._fields))
    part = _TRABAJADOR["part"]
    return part["orden"][1], part["pos_ord"][1], part["m_ord"][1], arbol


def arbol_local(paso):
    """Modo local: reconstruye el árbol UNA vez por paso (los demás trozos lo reutilizan)."""
    if _TRABAJADOR["paso_local"] != paso:
        part = _TRABAJADOR["part"]
        # SINCRONIZACIÓN: se lee pos mientras el principal espera en imap_unordered; el
        # principal solo escribe pos entre pasos, así que nadie la cambia durante la lectura.
        _TRABAJADOR["local"] = bh.preparar_arbol(part["pos"][1], part["m"][1], dict.fromkeys(FASES, 0.0))
        _TRABAJADOR["paso_local"] = paso
    return _TRABAJADOR["local"]


def pid_tras_pausa(_):
    """Tarea vacía: la pausa obliga a que la tomen procesos distintos."""
    time.sleep(0.05)
    return os.getpid()


def esperar_trabajadores(pool, k):
    """Vuelve cuando los k procesos terminaron iniciar_trabajador (una tarea solo corre
    después del inicializador). Así crear el Pool no se mezcla con la simulación."""
    vistos = set()
    for _ in range(50):
        vistos.update(pool.map(pid_tras_pausa, range(k), chunksize=1))
        if len(vistos) == k:
            return


def tarea_trozo(tarea):
    """Fuerzas de las partículas ordenadas [ini, fin). Devuelve (pid, s árbol, s fuerzas, interacciones)."""
    ini, fin, paso, descr_arbol, n_nodos, theta, eps = tarea
    inicio = time.perf_counter()
    if _TRABAJADOR["modo"] == "shared":
        orden, pos_ord, m_ord, arbol = arbol_compartido(descr_arbol, n_nodos)
    else:
        orden, pos_ord, m_ord, arbol = arbol_local(paso)
    medio = time.perf_counter()
    acc, interacciones = bh.aceleraciones_rango(ini, fin, arbol, pos_ord, m_ord, theta, eps)
    # SINCRONIZACIÓN: dos trozos nunca escriben la misma fila de acc, porque `orden` es una
    # permutación y los rangos [ini, fin) no se tocan. Por eso no hace falta candado.
    _TRABAJADOR["part"]["acc"][1][orden[ini:fin]] = acc
    return os.getpid(), medio - inicio, time.perf_counter() - medio, int(interacciones.sum())


# ----------------------------------------------------------------- programa

def leer_argumentos():
    p = argparse.ArgumentParser(description="Barnes-Hut con multiprocessing (S3)")
    p.add_argument("--input", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--dt", type=float, default=bh.DT)
    p.add_argument("--theta", type=float, default=bh.THETA)
    p.add_argument("--eps", type=float, default=bh.EPS)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tree", choices=["shared", "local"], default=MODO_ARBOL)
    p.add_argument("--out", required=True)
    return p.parse_args()


def simular(pool, comp, vel, a, tiempos, registro):
    """Leapfrog KDK; el principal integra sobre la memoria compartida entre pasos."""
    pos, acc = comp["part"]["pos"][1], comp["part"]["acc"][1]
    fuerzas_paralelas(pool, comp, 0, a, tiempos, registro)
    acc0 = acc.copy()
    for paso in range(1, a.steps + 1):
        inicio = time.perf_counter()
        bh.medio_kick(vel, acc, a.dt)
        bh.drift(pos, vel, a.dt)  # aquí ningún trabajador está leyendo pos (ver la barrera)
        tiempos["integracion"] += time.perf_counter() - inicio
        fuerzas_paralelas(pool, comp, paso, a, tiempos, registro)
        inicio = time.perf_counter()
        bh.medio_kick(vel, acc, a.dt)
        tiempos["integracion"] += time.perf_counter() - inicio
    return acc0


def escribir_workers(prefijo, registro):
    with open(f"{prefijo}_workers.csv", "w", encoding="utf-8") as f:
        f.write("trabajador,segundos,interacciones,trozos\n")
        for fila in sorted(registro.values(), key=lambda r: r["trabajador"]):
            f.write(f"{fila['trabajador']},{fila['segundos']:.6f},{fila['interacciones']},{fila['trozos']}\n")


def correr(a, pos0, vel, m, tiempos):
    """Crea la memoria compartida y el Pool UNA vez, simula y libera todo al final."""
    comp = {"part": bloques_particulas(len(pos0), a.tree), "arbol": {}, "capacidad": 0}
    try:
        comp["part"]["pos"][1][:] = pos0
        comp["part"]["m"][1][:] = m
        comp["capacidad"] = 2 * len(pos0) + 64
        comp["arbol"] = bloques_arbol(comp["capacidad"]) if a.tree == "shared" else {}
        inicio = time.perf_counter()
        with Pool(a.workers, initializer=iniciar_trabajador,
                  initargs=(a.tree, describir(comp["part"]))) as pool:
            esperar_trabajadores(pool, a.workers)
            tiempos["crear_pool"] = time.perf_counter() - inicio
            registro = {}
            inicio = time.perf_counter()
            acc0 = simular(pool, comp, vel, a, tiempos, registro)
            tiempos["simulacion"] = time.perf_counter() - inicio
        pos_final = comp["part"]["pos"][1].copy()
    finally:  # SINCRONIZACIÓN: el Pool ya terminó; recién ahora se puede borrar la memoria
        liberar(comp["part"])
        liberar(comp["arbol"])
    return acc0, pos_final, registro


def main():
    a = leer_argumentos()
    tiempos = dict.fromkeys(FASES, 0.0)
    inicio = time.perf_counter()
    pos, vel, m = leer_particulas(a.input)
    tiempos["lectura"] = time.perf_counter() - inicio
    # Compilar (o cargar de la caché) en el principal ANTES de crear el Pool: así los
    # trabajadores encuentran la caché escrita y nadie compila dentro de la medición.
    bh.calentar_numba(pos, m, a.theta, a.eps)
    acc0, pos_final, registro = correr(a, pos, vel, m, tiempos)
    # total = lectura + simulación, como en S2; crear el Pool se reporta aparte.
    tiempos["total"] = tiempos["lectura"] + tiempos.pop("simulacion")
    escribir_salidas(a.out, acc0, pos_final, vel, m, tiempos)
    with open(f"{a.out}_times.csv", "a", encoding="utf-8") as f:
        f.write(f"crear_pool,{tiempos['crear_pool']:.6f}\n")
    escribir_workers(a.out, registro)


if __name__ == "__main__":
    main()
