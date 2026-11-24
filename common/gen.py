"""Generador de entradas del contrato (ver CONTRATO.md).

Uso:  python common/gen.py --dist clusters --n 20000 --vel 0.3 --masas aleatorias
Salida: data/<dist>_N<N>_s<semilla>[_v][_m][_e].txt  (N filas: x y z vx vy vz m)
"""
import argparse
from pathlib import Path

import numpy as np

# Se escribe en <repo>/data sin importar desde dónde se llame, para que todos
# encuentren los archivos con el mismo nombre del contrato.
CARPETA_DATOS = Path(__file__).resolve().parent.parent / "data"

p = argparse.ArgumentParser()
p.add_argument("--dist", default="uniform", choices=["uniform", "clusters", "corner"])
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--vel", type=float, default=0.0)       # 0 = reposo; >0 = velocidades aleatorias
p.add_argument("--masas", default="iguales", choices=["iguales", "aleatorias"])
p.add_argument("--escala", type=float, default=1.0)    # lado del cubo; prueba posiciones fuera de [0, 1]
a = p.parse_args()

# Semilla fija: los archivos de datos no se suben, se regeneran idénticos con ella.
rng = np.random.default_rng(a.seed)
if a.dist == "uniform":            # requisito del enunciado
    pos = rng.random((a.n, 3))
elif a.dist == "clusters":         # demo: dos cúmulos densos que chocan
    h = a.n // 2
    pos = np.vstack([rng.normal(0.3, 0.03, (h, 3)), rng.normal(0.7, 0.03, (a.n - h, 3))])
else:                              # demo: casi todo en un rincón (castiga el reparto por espacio)
    k = int(a.n * 0.95)
    pos = np.vstack([rng.random((k, 3)) * 0.05, rng.random((a.n - k, 3))])
pos *= a.escala
vel = rng.normal(0.0, a.vel, (a.n, 3)) if a.vel > 0 else np.zeros((a.n, 3))
if a.dist == "clusters" and a.vel > 0:   # que se acerquen: uno a la derecha, otro a la izquierda
    vel[: a.n // 2, 0] += a.vel
    vel[a.n // 2 :, 0] -= a.vel
m = rng.random((a.n, 1)) if a.masas == "aleatorias" else np.ones((a.n, 1))
m /= m.sum()                       # masa total = 1: la energía queda en la misma escala para todo N
suf = ("_v" if a.vel > 0 else "") + ("_m" if a.masas == "aleatorias" else "") + ("_e" if a.escala != 1 else "")
CARPETA_DATOS.mkdir(exist_ok=True)
salida = CARPETA_DATOS / f"{a.dist}_N{a.n}_s{a.seed}{suf}.txt"
np.savetxt(salida, np.hstack([pos, vel, m]))
print(f"escrito {salida.relative_to(CARPETA_DATOS.parent)} ({a.n} partículas)")
