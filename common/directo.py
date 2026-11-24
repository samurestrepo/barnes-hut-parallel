"""Versión de referencia con suma directa O(N²) y el CLI del contrato.

No es una de las 4 versiones: sirve para probar el verificador.
  - Tal cual, sus salidas deben dar error ≈ 0 en todas las pruebas.
  - Con --sin-segundo-kick (bug típico del leapfrog) el verificador debe decir FALLA.
θ y --workers se aceptan para cumplir el CLI, pero se ignoran: la suma directa
es lo mismo que Barnes-Hut con θ = 0, y corre en un solo proceso.

  python common/directo.py --input data/uniform_N2000_s42.txt --steps 10 --out /tmp/ref
"""
import argparse
import time

import numpy as np

from fisica import aceleraciones_directas, leer_particulas

FASES = ["lectura", "construccion", "ordenamiento", "recorrido", "integracion", "comunicacion", "total"]
FORMATO = "%.17g"  # 17 cifras: el texto no debe agregar error de redondeo a float64


def leer_argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--theta", type=float, default=0.5)   # ignorado
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=1)     # ignorado
    p.add_argument("--out", required=True)
    p.add_argument("--sin-segundo-kick", action="store_true",
                   help="bug a propósito: omite el medio kick final de cada paso")
    return p.parse_args()


def simular(pos, vel, m, a, tiempos):
    """Leapfrog KDK; devuelve las aceleraciones del paso 0 para acc0.txt."""
    inicio = time.perf_counter()
    acc = aceleraciones_directas(pos, m, a.eps)
    tiempos["recorrido"] += time.perf_counter() - inicio
    acc0 = acc.copy()
    for _ in range(a.steps):
        inicio = time.perf_counter()
        vel += 0.5 * a.dt * acc
        pos += a.dt * vel
        tiempos["integracion"] += time.perf_counter() - inicio
        inicio = time.perf_counter()
        acc = aceleraciones_directas(pos, m, a.eps)
        tiempos["recorrido"] += time.perf_counter() - inicio
        if not a.sin_segundo_kick:
            vel += 0.5 * a.dt * acc
    return acc0


def escribir_salidas(prefijo, acc0, pos, vel, m, tiempos):
    np.savetxt(f"{prefijo}_acc0.txt", acc0, fmt=FORMATO)
    np.savetxt(f"{prefijo}_final.txt", np.column_stack([pos, vel, m]), fmt=FORMATO)
    with open(f"{prefijo}_times.csv", "w", encoding="utf-8") as f:
        f.write("fase,segundos\n")
        for fase in FASES:
            f.write(f"{fase},{tiempos[fase]:.6f}\n")


def main():
    a = leer_argumentos()
    tiempos = dict.fromkeys(FASES, 0.0)
    inicio_total = time.perf_counter()
    pos, vel, m = leer_particulas(a.input)
    tiempos["lectura"] = time.perf_counter() - inicio_total
    acc0 = simular(pos, vel, m, a, tiempos)
    tiempos["total"] = time.perf_counter() - inicio_total
    escribir_salidas(a.out, acc0, pos, vel, m, tiempos)


if __name__ == "__main__":
    main()
