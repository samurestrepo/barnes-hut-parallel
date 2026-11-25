"""Gráficas de escalado (tarea S5): tiempo, speedup, eficiencia y Karp-Flatt.

  python common/plot.py --out results/mp/uniform_N20000 --base 3.315 \
      --serie shared=results/mp/uniform_shared_bench.csv --serie local=results/mp/uniform_local_bench.csv

Cada --serie es un *_bench.csv de bench.py (se promedian las repeticiones).
--base es el tiempo secuencial T1 (p. ej. el total de S2). Si se omite, cada
serie usa su propio tiempo con 1 trabajador.
Escribe <out>_escalado.png (4 paneles) y <out>_escalado.csv.

  speedup     S(p) = T1 / T(p)
  eficiencia  E(p) = S(p) / p
  Karp-Flatt  e(p) = (1/S − 1/p) / (1 − 1/p): fracción secuencial "experimental".
              Si crece con p, el overhead (comunicación, núcleos compartidos) crece;
              si queda constante, lo que frena es la parte secuencial (Amdahl).
"""
import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo PNG
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categórica validada (dataviz, modo claro), en orden fijo: la serie 1 siempre azul.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]
TINTA, TINTA_SUAVE, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def leer_bench(ruta):
    """{trabajadores: total promedio} de un *_bench.csv."""
    por_k = defaultdict(list)
    with open(ruta, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            por_k[int(fila["trabajadores"])].append(float(fila["total"]))
    return {k: st.mean(v) for k, v in sorted(por_k.items())}


def metricas(tiempos, base):
    """Filas (p, T, S, E, e_KF) de una serie."""
    t1 = base if base else tiempos[min(tiempos)]
    filas = []
    for p, t in tiempos.items():
        s = t1 / t
        kf = (1 / s - 1 / p) / (1 - 1 / p) if p > 1 else float("nan")
        filas.append((p, t, s, s / p, kf))
    return filas


def estilo(ax, titulo, ylabel):
    ax.set_title(titulo, loc="left", fontsize=11, color=TINTA)
    ax.set_xlabel("trabajadores (p)", color=TINTA_SUAVE)
    ax.set_ylabel(ylabel, color=TINTA_SUAVE)
    ax.set_xscale("log", base=2)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_facecolor(FONDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA_SUAVE)


def etiquetar_sin_choque(ax, finales, separacion=12):
    """Nombre de cada serie al final de su línea; si dos quedan a menos de `separacion`
    puntos en vertical, se corren hacia arriba para que no se pisen."""
    ax.autoscale_view()
    puntos = sorted((ax.transData.transform((x, y))[1], nombre, x, y) for nombre, x, y in finales)
    ultimo = -1e9
    escala = ax.figure.dpi / 72.0  # píxeles por punto tipográfico
    for alto, nombre, x, y in puntos:
        destino = max(alto, ultimo + separacion * escala)
        ax.annotate(nombre, (x, y), xytext=(6, (destino - alto) / escala), textcoords="offset points",
                    color=TINTA_SUAVE, fontsize=9, va="center")
        ultimo = destino


def dibujar(series, out, base, nucleos):
    fig, ejes = plt.subplots(2, 2, figsize=(10, 7.5), facecolor=FONDO)
    paneles = [(1, "Tiempo total", "segundos"), (2, "Speedup", "S(p)"),
               (3, "Eficiencia", "E(p)"), (4, "Karp-Flatt", "fracción secuencial e(p)")]
    for (col, titulo, ylabel), ax in zip(paneles, ejes.ravel()):
        estilo(ax, titulo, ylabel)
        finales = []
        for i, (nombre, filas) in enumerate(series.items()):
            xs = [f[0] for f in filas if not (col == 4 and f[0] == 1)]
            ys = [f[col] for f in filas if not (col == 4 and f[0] == 1)]
            ax.plot(xs, ys, color=COLORES[i % len(COLORES)], linewidth=2, marker="o", markersize=7, label=nombre)
            finales.append((nombre, xs[-1], ys[-1]))
        if col == 2:  # referencia: speedup ideal S = p (antes de etiquetar: cambia la escala)
            ps = sorted({f[0] for filas in series.values() for f in filas})
            ax.plot(ps, ps, color=TINTA_SUAVE, linestyle="--", linewidth=1, label="ideal S = p")
        etiquetar_sin_choque(ax, finales)
        if nucleos:
            ax.axvline(nucleos, color=TINTA_SUAVE, linestyle=":", linewidth=1)
        ax.set_xticks(sorted({f[0] for filas in series.values() for f in filas}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ejes[0, 1].legend(frameon=False, fontsize=9)
    nota = f"T1 = {base:.3f} s (secuencial)" if base else "T1 = tiempo con 1 trabajador"
    if nucleos:
        nota += f" · línea punteada: {nucleos} núcleos físicos"
    fig.text(0.01, 0.005, nota, color=TINTA_SUAVE, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(f"{out}_escalado.png", dpi=130)


def escribir_csv(series, out):
    with open(f"{out}_escalado.csv", "w", encoding="utf-8") as f:
        f.write("serie,trabajadores,total_s,speedup,eficiencia,karp_flatt\n")
        for nombre, filas in series.items():
            for p, t, s, e, kf in filas:
                f.write(f"{nombre},{p},{t:.4f},{s:.3f},{e:.3f},{kf:.4f}\n")


def main():
    p = argparse.ArgumentParser(description="Gráficas de escalado (S5)")
    p.add_argument("--serie", action="append", required=True, help="nombre=ruta/a/X_bench.csv")
    p.add_argument("--out", required=True, help="prefijo de salida, p. ej. results/mp/uniform_N20000")
    p.add_argument("--base", type=float, help="tiempo secuencial T1 en segundos")
    p.add_argument("--nucleos", type=int, help="núcleos físicos (línea de referencia)")
    a = p.parse_args()
    series = {}
    for par in a.serie:
        nombre, ruta = par.split("=", 1)
        series[nombre] = metricas(leer_bench(ruta), a.base)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    escribir_csv(series, a.out)
    dibujar(series, a.out, a.base, a.nucleos)
    print(f"escrito {a.out}_escalado.png y {a.out}_escalado.csv")


if __name__ == "__main__":
    main()
