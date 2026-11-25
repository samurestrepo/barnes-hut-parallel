# Ensayo de demo (2026-11-24)

N = 20 000, dt = 2.5e-4, θ = 0.5, ε = 0.01, **100 pasos**, 4 trabajadores, máquina i5-6300U (2 núcleos, 4 hilos).
Datos: `clusters` = `gen.py --dist clusters --n 20000 --vel 0.3 --masas aleatorias`; `corner` = `gen.py --dist corner --n 20000`.
Deriva = `python common/verify.py deriva --input <datos> --final <prefijo>_final.txt`.

| Datos | Versión | Total (s) | Recorrido (s) | Comunicación (s) | Deriva de energía | Trabajadores: segundos (mín – máx) |
| --- | --- | --- | --- | --- | --- | --- |
| clusters | S2 secuencial | 29.02 | 28.06 | 0 | 1.273e-2 | — |
| clusters | mp `--tree shared` w4 | 15.91 | 14.76 | 0.03 | 1.273e-2 | 13.67 – 13.80 |
| clusters | MPI `cantidad` n4 | 16.62 | 13.32 | 1.96 | 1.278e-2 | 13.02 – 13.32 |
| corner | S2 secuencial | 25.15 | 24.21 | 0 | 4.567e-4 | — |
| corner | mp `--tree shared` w4 | 12.76 | 11.66 | 0.02 | 4.567e-4 | 11.11 – 11.17 |
| corner | MPI `cantidad` n4 | 14.08 | 11.42 | 4.91 | 7.612e-4 | 7.08 – 11.42 |

Fuentes: `results/demo/<datos>_<versión>_times.csv` y `_workers.csv`.

**Qué mirar:**
- **Corrección**: S2 y mp dan la misma deriva (mismo árbol, mismas cuentas); MPI difiere un poco porque su árbol combinado aproxima distinto. Con `clusters` N = 20 000 la deriva (1.27e-2) es mayor que con N = 2000 (8.9e-3). La masa total de cada cúmulo es la misma (0.5) en los dos casos; no se midió por qué crece. Hipótesis sin comprobar: con 10 veces más partículas hay más pares a distancias del orden de ε = 0.01, donde el error de θ y el del paso de tiempo pesan más. Queda bajo el umbral propuesto (2e-2), pero no bajo el actual (1e-3).
- **Balanceo con `corner` en 100 pasos**: mp, con trozos dinámicos, queda dentro del 1% (11.11–11.17 s). MPI con `cantidad` (reparto estático por número de partículas) se desbalancea a medida que el rincón colapsa: el rango 3 tarda 7.08 s y el 1 tarda 11.42 s (239 frente a 439 millones de interacciones). Los demás esperan al más lento: por eso MPI tiene 4.91 s de comunicación. A los 20 pasos esto casi no se veía (1.60–1.79 s): **"cantidad no es trabajo"** crece con el tiempo de simulación.
- **Speedup del ensayo**: mp 1.82 (clusters) y 1.97 (corner); MPI 1.75 y 1.79, contra S2.
