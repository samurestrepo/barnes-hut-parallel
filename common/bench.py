"""Bucle de mediciones (tarea S5): corre una versión con 1, 2, 4, 8… trabajadores.

  python common/bench.py --impl mp --etiqueta uniform_shared --cmd "python py_mp/bh_mp.py --tree shared"
  python common/bench.py --impl mpi --etiqueta uniform --cmd "python mpi/bh_mpi.py" --lanzador "mpiexec -n {k}"
  python common/bench.py --impl omp --etiqueta uniform --cmd "./c_omp/bh_omp"          # Sebastián
  python common/bench.py --impl cuda --etiqueta bloques --cmd "./cuda/bh_cuda" --trabajadores 64,128,256

Por defecto los trabajadores son las potencias de 2 hasta 2× los núcleos lógicos
(aquí: 1, 2, 4, 8), para ver qué pasa al sobresuscribir. Con MPI, `--lanzador`
pone delante el `mpiexec -n {k}` (o `mpirun --oversubscribe -n {k}` con OpenMPI)
y `--workers` se pasa igual, aunque MPI lo ignore.

Escribe en results/<impl>/:
  bench_<etiqueta>/w<k>_r<rep>_times.csv (y lo demás que escriba la versión; en una
  subcarpeta propia para no pisar archivos de otras mediciones con el mismo nombre)
  <etiqueta>_bench.csv   trabajadores,repeticion,segundos de pared,fases de times.csv
y guarda la descripción del procesador en results/lscpu.txt.
"""
import argparse
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FASES = ["lectura", "construccion", "ordenamiento", "recorrido", "integracion", "comunicacion", "total"]


def trabajadores_por_defecto():
    """1, 2, 4, … hasta 2× los núcleos lógicos (incluido si no es potencia de 2)."""
    tope = 2 * (os.cpu_count() or 1)
    lista, k = [], 1
    while k <= tope:
        lista.append(k)
        k *= 2
    return lista if tope in lista else lista + [tope]


def guardar_lscpu():
    """`lscpu` en Linux/WSL; en Windows, lo equivalente con wmic. Para el informe."""
    ruta = RAIZ / "results" / "lscpu.txt"
    for cmd in (["lscpu"], ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize", "/format:list"]):
        try:
            texto = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            break
        except (OSError, subprocess.CalledProcessError):
            texto = ""
    cabecera = f"# {platform.platform()} | Python {platform.python_version()} | os.cpu_count() = {os.cpu_count()}\n"
    lineas = [linea for linea in texto.replace("\r", "").splitlines() if linea.strip()]  # wmic deja líneas vacías
    ruta.write_text(cabecera + "\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def leer_tiempos(prefijo):
    """times.csv → {fase: valor}; también las filas extra (bytes, crear_pool)."""
    filas = Path(f"{prefijo}_times.csv").read_text(encoding="utf-8").split()[1:]
    return {f.split(",")[0]: float(f.split(",")[1]) for f in filas}


def correr_una(a, k, rep):
    """Una corrida con k trabajadores. Devuelve (segundos de pared, tiempos) o None si falla."""
    prefijo = RAIZ / "results" / a.impl / f"bench_{a.etiqueta}" / f"w{k}_r{rep}"
    lanzador = shlex.split(a.lanzador.format(k=k)) if a.lanzador else []
    cmd = lanzador + shlex.split(a.cmd) + [
        "--input", a.input, "--steps", str(a.steps), "--dt", str(a.dt), "--theta", str(a.theta),
        "--eps", str(a.eps), "--workers", str(k), "--out", str(prefijo)]
    inicio = time.perf_counter()
    proceso = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, timeout=a.limite)
    pared = time.perf_counter() - inicio
    if proceso.returncode != 0:
        print(f"  k={k}: FALLÓ ({proceso.returncode}) {proceso.stderr.strip()[-300:]}", flush=True)
        return None
    return pared, leer_tiempos(prefijo)


def leer_argumentos():
    p = argparse.ArgumentParser(description="Bucle de mediciones (S5)")
    p.add_argument("--cmd", required=True, help='versión, p. ej. "python py_mp/bh_mp.py --tree shared"')
    p.add_argument("--impl", required=True, help="subcarpeta de results/: mp, mpi, omp, cuda…")
    p.add_argument("--etiqueta", required=True, help="nombre de la serie, p. ej. uniform_shared")
    p.add_argument("--input", default="data/uniform_N20000_s42.txt")
    p.add_argument("--steps", type=int, default=20)  # contrato: 20 pasos para medir
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--trabajadores", help="lista separada por comas (por defecto potencias de 2 hasta 2× lógicos)")
    p.add_argument("--lanzador", default="", help='p. ej. "mpiexec -n {k}" o "mpirun --oversubscribe -n {k}"')
    p.add_argument("--repeticiones", type=int, default=1)
    p.add_argument("--limite", type=float, default=3600, help="segundos máximos por corrida")
    return p.parse_args()


def main():
    a = leer_argumentos()
    lista = [int(k) for k in a.trabajadores.split(",")] if a.trabajadores else trabajadores_por_defecto()
    (RAIZ / "results" / a.impl / f"bench_{a.etiqueta}").mkdir(parents=True, exist_ok=True)
    print(f"procesador guardado en {guardar_lscpu().relative_to(RAIZ)}; trabajadores {lista}")
    resumen = RAIZ / "results" / a.impl / f"{a.etiqueta}_bench.csv"
    with open(resumen, "w", encoding="utf-8") as f:
        f.write("trabajadores,repeticion,pared," + ",".join(FASES) + "\n")
        for k in lista:
            for rep in range(1, a.repeticiones + 1):
                r = correr_una(a, k, rep)
                if r is None:
                    continue
                pared, t = r
                f.write(f"{k},{rep},{pared:.4f}," + ",".join(f"{t.get(fa, 0.0):.6f}" for fa in FASES) + "\n")
                f.flush()
                print(f"  k={k} rep={rep}: total {t['total']:.3f} s (pared {pared:.1f} s)", flush=True)
    print(f"resumen en {resumen.relative_to(RAIZ)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
