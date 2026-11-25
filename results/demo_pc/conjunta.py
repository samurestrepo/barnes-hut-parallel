"""Gráfica conjunta: las 3 series de Python (S3 shared, S3 local, S4 MPI) y OpenMP.

common/plot.py usa un solo T1 para todas las series. Aquí cada serie lleva el
suyo, porque el secuencial de Python (S2) y el de C (C1) son programas distintos:
el speedup de cada versión se mide contra SU secuencial. Reutiliza las funciones
de common/plot.py (no lo modifica).

  python results/demo_pc/conjunta.py --out results/demo_pc/conjunta_N20000 --nucleos 8 \
      --serie "mp shared=results/demo_pc/mp_shared/uniform_N20000_bench.csv:4.1" ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
import plot  # noqa: E402

# Perilla: 4 colores (plot.py trae 3 y la cuarta serie repetiría el azul).
plot.COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a4fd0"]


def main():
    p = argparse.ArgumentParser(description="Escalado conjunto con un T1 por serie")
    p.add_argument("--serie", action="append", required=True, help="nombre=ruta_bench.csv:T1")
    p.add_argument("--out", required=True)
    p.add_argument("--nucleos", type=int)
    a = p.parse_args()
    series = {}
    for par in a.serie:
        nombre, resto = par.split("=", 1)
        ruta, base = resto.rsplit(":", 1)
        series[nombre] = plot.metricas(plot.leer_bench(ruta), float(base))
    plot.escribir_csv(series, a.out)
    plot.dibujar(series, a.out, None, a.nucleos)
    # dibujar() escribe al pie "T1 = tiempo con 1 trabajador", que aquí no es cierto:
    # se reemplaza ese texto por los T1 reales y se guarda otra vez.
    fig = plot.plt.gcf()
    bases = ", ".join(f"{par.split('=', 1)[0]}: {par.rsplit(':', 1)[1]} s" for par in a.serie)
    fig.texts[-1].set_text(f"T1 por serie ({bases})"
                           + (f" · línea punteada: {a.nucleos} núcleos físicos" if a.nucleos else ""))
    fig.savefig(f"{a.out}_escalado.png", dpi=130)
    print(f"escrito {a.out}_escalado.png y {a.out}_escalado.csv")


if __name__ == "__main__":
    main()
