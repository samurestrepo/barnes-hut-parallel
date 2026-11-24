"""Verificador común: decide si una versión cumple CONTRATO.md.

Tres modos:

  # Corre la versión con las 5 pruebas del contrato e imprime PASA/FALLA por prueba
  python common/verify.py correr --cmd "python py_core/bh_core.py"
  python common/verify.py correr --cmd "mpirun -n 4 python mpi/bh_mpi.py" --solo trayectoria
  python common/verify.py correr --cmd "./cuda/bh_cuda" --float32 --workers 128

  # Revisa salidas ya escritas (p. ej. en Colab), sin correr nada
  python common/verify.py revisar --prueba fuerzas_bh --salida results/cuda/fbh

  # Deriva de energía de cualquier corrida (incluida la distribución de la demo)
  python common/verify.py deriva --input data/X.txt --final PREFIJO_final.txt

Para probar el verificador mismo: --cmd "python common/directo.py" debe PASAR
todo, y --cmd "python common/directo.py --sin-segundo-kick" debe FALLAR.
"""
import argparse
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from fisica import energia_total, leapfrog_directo, aceleraciones_directas, leer_particulas

RAIZ = Path(__file__).resolve().parent.parent
EPS = 0.01  # ε del contrato; las pruebas no lo cambian

# Las 5 pruebas del contrato. Cada una corre la versión una vez con estos datos y
# parámetros, y mide una sola cosa. `umbral` = (float64, float32 para CUDA).
DATOS_UNIFORM = ("uniform_N2000_s42.txt", ["--dist", "uniform", "--n", "2000"])
DATOS_MOVIMIENTO = ("clusters_N2000_s42_v_m.txt",
                    ["--dist", "clusters", "--n", "2000", "--vel", "0.3", "--masas", "aleatorias"])
PRUEBAS = {
    "fuerzas_directa": dict(datos=DATOS_UNIFORM, theta=0.0, dt=1e-3, pasos=1,
                            medida="fuerzas_max", umbral=(1e-6, 1e-3)),
    "fuerzas_bh": dict(datos=DATOS_UNIFORM, theta=0.5, dt=1e-3, pasos=1,
                       medida="fuerzas_mediana", umbral=(1e-2, 1e-2)),
    "trayectoria": dict(datos=DATOS_UNIFORM, theta=0.0, dt=1e-3, pasos=10,
                        medida="trayectoria", umbral=(1e-6, 1e-3)),
    "energia_reposo": dict(datos=DATOS_UNIFORM, theta=0.0, dt=1e-3, pasos=100,
                           medida="energia", umbral=(1e-5, 1e-3)),
    # Suposición del diseño (riesgo 3): umbral a ajustar con S2 y dejar fijo.
    "energia_movimiento": dict(datos=DATOS_MOVIMIENTO, theta=0.5, dt=2.5e-4, pasos=100,
                               medida="energia", umbral=(1e-3, 1e-3)),
}
FASES = ["lectura", "construccion", "ordenamiento", "recorrido", "integracion", "comunicacion", "total"]


class SalidaInvalida(Exception):
    """La versión no escribió lo que pide el contrato: la prueba FALLA sin medir."""


# ----------------------------------------------------------------- datos y salidas

def ruta_datos(prueba):
    """Devuelve el archivo de entrada de la prueba; lo genera con gen.py si falta."""
    nombre, args_gen = prueba["datos"]
    ruta = RAIZ / "data" / nombre
    if not ruta.exists():
        subprocess.run([sys.executable, str(RAIZ / "common" / "gen.py"), *args_gen], check=True)
    return ruta


def leer_salida(ruta, columnas, n):
    """Lee una salida N×columnas y comprueba la forma antes de comparar nada."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise SalidaInvalida(f"no existe {ruta.name}")
    datos = np.loadtxt(ruta, ndmin=2)
    if datos.shape != (n, columnas):
        raise SalidaInvalida(f"{ruta.name} tiene forma {datos.shape}, se esperaba {(n, columnas)}")
    if not np.all(np.isfinite(datos)):
        raise SalidaInvalida(f"{ruta.name} tiene NaN o infinitos")
    return datos


def avisos_de_tiempos(prefijo):
    """times.csv no decide PASA/FALLA, pero se avisa si no sigue el contrato."""
    ruta = Path(f"{prefijo}_times.csv")
    if not ruta.exists():
        return [f"falta {ruta.name}"]
    lineas = ruta.read_text(encoding="utf-8").split()
    fases = {linea.split(",")[0] for linea in lineas[1:]}
    faltan = [f for f in FASES if f not in fases]
    return [f"{ruta.name} sin fases: {', '.join(faltan)}"] if faltan else []


# ----------------------------------------------------------------- medidas

def error_relativo_fuerzas(entrada, prefijo):
    """Error relativo de la aceleración de cada partícula frente a la suma directa."""
    pos, _, m = leer_particulas(entrada)
    acc = leer_salida(f"{prefijo}_acc0.txt", 3, len(pos))
    ref = aceleraciones_directas(pos, m, EPS)
    return np.linalg.norm(acc - ref, axis=1) / np.linalg.norm(ref, axis=1)


def medir_fuerzas_max(entrada, prefijo, prueba):
    # θ = 0: Barnes-Hut abre todas las celdas y debe dar la suma directa salvo redondeo.
    return error_relativo_fuerzas(entrada, prefijo).max(), "error relativo máximo de a"


def medir_fuerzas_mediana(entrada, prefijo, prueba):
    # θ = 0.5: aproximar es lo esperado; la mediana ignora las pocas partículas con
    # aceleración casi nula, cuyo error relativo se dispara aunque el código esté bien.
    return np.median(error_relativo_fuerzas(entrada, prefijo)), "mediana error relativo de a"


def medir_trayectoria(entrada, prefijo, prueba):
    # Se compara posición por posición (no la energía) porque así también se detecta
    # una salida en orden Morton en vez del orden de la entrada.
    pos, vel, m = leer_particulas(entrada)
    final = leer_salida(f"{prefijo}_final.txt", 7, len(pos))
    pos_ref, _ = leapfrog_directo(pos, vel, m, prueba["dt"], EPS, prueba["pasos"])
    # Posiciones de orden 1 (cubo unitario): el error absoluto ya es relativo a la caja.
    return np.linalg.norm(final[:, 0:3] - pos_ref, axis=1).max(), "error máximo de posición"


def deriva_energia(entrada, ruta_final):
    """|E_final − E_inicial| / |E_inicial|, con la energía de cada archivo."""
    pos, vel, m = leer_particulas(entrada)
    final = leer_salida(ruta_final, 7, len(pos))
    e0 = energia_total(pos, vel, m, EPS)
    ef = energia_total(final[:, 0:3], final[:, 3:6], final[:, 6], EPS)
    return abs(ef - e0) / abs(e0), e0, ef


def medir_energia(entrada, prefijo, prueba):
    deriva, _, _ = deriva_energia(entrada, f"{prefijo}_final.txt")
    return deriva, f"deriva de energía ({prueba['pasos']} pasos)"


MEDIDAS = {"fuerzas_max": medir_fuerzas_max, "fuerzas_mediana": medir_fuerzas_mediana,
           "trayectoria": medir_trayectoria, "energia": medir_energia}


def revisar_prueba(nombre, prefijo, float32):
    """Mide una prueba sobre salidas ya escritas. Devuelve (pasa, valor, umbral, texto)."""
    prueba = PRUEBAS[nombre]
    umbral = prueba["umbral"][1 if float32 else 0]
    try:
        valor, texto = MEDIDAS[prueba["medida"]](ruta_datos(prueba), prefijo, prueba)
    except SalidaInvalida as e:
        return False, float("nan"), umbral, str(e)
    return bool(valor < umbral), valor, umbral, texto


# ----------------------------------------------------------------- ejecución

def ejecutar_version(cmd, nombre, prefijo, workers):
    """Corre la versión con el CLI del contrato. Devuelve (ok, segundos, mensaje)."""
    prueba = PRUEBAS[nombre]
    argumentos = shlex.split(cmd) + [
        "--input", str(ruta_datos(prueba)), "--steps", str(prueba["pasos"]),
        "--dt", str(prueba["dt"]), "--theta", str(prueba["theta"]), "--eps", str(EPS),
        "--workers", str(workers), "--out", str(prefijo)]
    inicio = time.perf_counter()
    proceso = subprocess.run(argumentos, cwd=RAIZ, capture_output=True, text=True)
    segundos = time.perf_counter() - inicio
    if proceso.returncode != 0:
        ultimas = (proceso.stderr or proceso.stdout).strip().splitlines()[-3:]
        return False, segundos, f"salió con código {proceso.returncode}: " + " | ".join(ultimas)
    return True, segundos, ""


def imprimir_fila(nombre, pasa, valor, umbral, texto):
    veredicto = "PASA" if pasa else "FALLA"
    print(f"{nombre:<20} {veredicto:<6} {valor:>10.2e} < {umbral:<8.0e} {texto}", flush=True)


def modo_correr(args):
    nombres = args.solo.split(",") if args.solo else list(PRUEBAS)
    carpeta = Path(args.carpeta or tempfile.mkdtemp(prefix="verify_"))
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"versión: {args.cmd}\nsalidas en: {carpeta}\n")
    todas = True
    for nombre in nombres:
        prefijo = carpeta / nombre
        ok, segundos, error = ejecutar_version(args.cmd, nombre, prefijo, args.workers)
        if ok:
            pasa, valor, umbral, texto = revisar_prueba(nombre, prefijo, args.float32)
            texto += f"  [{segundos:.1f} s]"
            for aviso in avisos_de_tiempos(prefijo):
                texto += f"  AVISO: {aviso}"
        else:
            pasa, valor, umbral, texto = False, float("nan"), PRUEBAS[nombre]["umbral"][0], error
        imprimir_fila(nombre, pasa, valor, umbral, texto)
        todas &= pasa
    print("\nRESULTADO:", "PASA todo" if todas else "FALLA al menos una prueba")
    return 0 if todas else 1


def modo_revisar(args):
    pasa, valor, umbral, texto = revisar_prueba(args.prueba, args.salida, args.float32)
    imprimir_fila(args.prueba, pasa, valor, umbral, texto)
    return 0 if pasa else 1


def modo_deriva(args):
    global EPS
    EPS = args.eps
    deriva, e0, ef = deriva_energia(args.input, args.final)
    print(f"E inicial = {e0:.12e}\nE final   = {ef:.12e}\nderiva    = {deriva:.3e}")
    return 0


def leer_argumentos():
    p = argparse.ArgumentParser(description="Verificador del contrato Barnes-Hut")
    modos = p.add_subparsers(dest="modo", required=True)
    c = modos.add_parser("correr", help="corre la versión y aplica las pruebas")
    c.add_argument("--cmd", required=True, help='programa, p. ej. "python py_core/bh_core.py"')
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--float32", action="store_true", help="umbrales de CUDA en float32")
    c.add_argument("--solo", help="pruebas separadas por coma: " + ",".join(PRUEBAS))
    c.add_argument("--carpeta", help="dónde dejar las salidas (por defecto, una temporal)")
    r = modos.add_parser("revisar", help="revisa salidas ya escritas")
    r.add_argument("--prueba", required=True, choices=list(PRUEBAS))
    r.add_argument("--salida", required=True, help="PREFIJO con que se corrió la versión")
    r.add_argument("--float32", action="store_true")
    d = modos.add_parser("deriva", help="deriva de energía de cualquier corrida")
    d.add_argument("--input", required=True)
    d.add_argument("--final", required=True)
    d.add_argument("--eps", type=float, default=EPS)
    return p.parse_args()


if __name__ == "__main__":
    # Para que las tildes salgan bien también en la consola de Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    argumentos = leer_argumentos()
    modo = {"correr": modo_correr, "revisar": modo_revisar, "deriva": modo_deriva}[argumentos.modo]
    sys.exit(modo(argumentos))
