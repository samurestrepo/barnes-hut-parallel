# Barnes-Hut con MPI (mpi4py)

*Borrador sin pulir. Todos los números salen de `results/`; la fuente va entre paréntesis.*

## Diseño

`mpi/bh_mpi.py` reutiliza el árbol, el recorrido y el leapfrog de `py_core/bh_core.py`. En cada paso:

1. `Allreduce` de mínimos y máximos: todos usan la misma caja para los códigos de Morton.
2. El rango 0 junta códigos y partículas (`Gatherv`), las ordena, corta en P trozos y las reparte (`Scatterv`). Con `--particion cantidad` cada rango recibe N/P partículas contiguas en orden Morton; con `--particion espacio` el espacio de códigos se corta en P tramos de igual volumen (el reparto ingenuo).
3. Cada rango construye el árbol de su trozo, y con `Allgather` todos conocen la caja de todos.
4. **Poda:** cada rango recorre su árbol contra la caja de cada rango remoto. Una celda que cumple θ vista desde el punto más cercano de esa caja la cumple para cualquier partícula de ella y viaja como (centro de masa, masa); si no, se abre, y las hojas abiertas viajan como partículas. El intercambio es un `Alltoall` de cuentas más un `Alltoallv`.
5. Cada rango arma un árbol con sus partículas más lo recibido, calcula las aceleraciones de las suyas e integra.

Todas las colectivas usan búferes de NumPy y se llaman en el mismo orden en todos los rangos (11 por paso), así no puede haber deadlock por orden. No hay envíos punto a punto.

## Verificación

Con `mpiexec -n 1`, `2` y `4`: fuerzas con θ = 0, 1.5e-14; con θ = 0.5, mediana 3.2e-3 a 3.3e-3; trayectoria, 2.8e-17; energía en reposo, 3.4e-8. Con θ = 0 las salidas son idénticas bit a bit a la versión secuencial. La energía en movimiento da 8.9e-3 a 9.0e-3 frente al umbral provisional de 1e-3, el mismo caso que la versión secuencial (ver la sección de multiprocessing y `CONTRATO_DUDAS.md`).

## Comunicación: ramas podadas frente al árbol completo

Bytes recibidos en el `Alltoallv`, sumados sobre rangos y pasos, frente a lo que llegaría si cada rango mandara su árbol entero (todos sus nodos y partículas, 32 bytes cada uno). Fuente: filas `bytes_*` de times.csv en `results/mpi/uniform_N20000_cantidad_n*_r1_times.csv` y `results/mpi/bench_uniform_N200000_cantidad/`.

| Rangos | N = 20 000 | N = 200 000 |
| --- | --- | --- |
| 2 | 5.8 MB de 17.6 MB (32.8%) | 27.9 MB de 181.0 MB (15.4%) |
| 4 | 16.3 MB de 52.9 MB (30.9%) | 95.8 MB de 543.0 MB (17.7%) |
| 8 | 33.4 MB de 123.4 MB (27.1%) | 197.7 MB de 1267.2 MB (15.6%) |

Con más partículas la proporción baja: las celdas lejanas resumen más partículas.

## Rendimiento

Base: la versión secuencial, 3.315 s (N = 20 000) y 52.38 s (N = 200 000), 20 pasos, θ = 0.5; máquina de 2 núcleos físicos y 4 lógicos. Gráficas con las tres versiones de Python: `results/mpi/uniform_N20000_escalado.png` y `results/mpi/uniform_N200000_escalado.png`.

| N | S(1) | S(2) | S(4) | S(8) |
| --- | --- | --- | --- | --- |
| 20 000 | 0.88 | 1.36 | 1.66 | 1.29 |
| 200 000 | 0.97 | 1.43 | 1.95 | 1.94 |

(`results/mpi/uniform_N*_escalado.csv`)

- **Amdahl.** La fracción secuencial medida en la versión secuencial (ordenamiento y lectura, ≈ 5.7%) daba como techo S(4) = 3.42. Se midió 1.66. La diferencia es sobre todo el hardware (2 núcleos físicos) y luego la comunicación. Con N = 200 000, la comunicación (incluida la espera en colectivas) sube de 0.17 s con 1 rango a 1.66, 2.12 y 3.54 s con 2, 4 y 8, mientras el ordenamiento del rango 0 se queda en ≈ 0.33 s (`results/mpi/uniform_N200000_cantidad_bench.csv`).
- **Con 1 rango es 14% más lenta que la secuencial** (3.78 frente a 3.315 s): arma dos árboles por paso (el local y el combinado) y pasa igual por las colectivas.
- **Espera activa de Microsoft MPI.** En la primera medición, 8 rangos tardaban 9.6 s (speedup 0.35) con 8.0 s de "comunicación", aunque cada rango calculaba ≈ 1 s (`results/mpi/sin_limite_giro/`). Hipótesis: con 8 procesos en 4 hilos de hardware, los rangos que esperan dentro de una colectiva siguen consultando (espera activa) y le quitan CPU a los que calculan. No se midió directamente; lo que sí se midió es que el límite de espera activa lo corrige. Con `MPICH_PROGRESS_SPIN_LIMIT=16` (fijado arriba del archivo) bajó a 2.27 s (`results/mpi/resumen_uniform_N20000.csv`).

## Reparto: cantidad frente a espacio

`corner`, N = 20 000, dt = 2.5e-4, 4 rangos, 20 pasos (`results/mpi/corner_N20000_*_n4_workers.csv`):

| Partición | Total | Partículas por rango | Recorrido por rango |
| --- | --- | --- | --- |
| cantidad | 2.40 s | 5000 cada uno | 1.60 a 1.79 s |
| espacio | 4.75 s | 19 254 / 241 / 252 / 253 | 4.16 / 0.03 / 0.03 / 0.03 s |

Con `espacio`, el rincón denso cae entero en el primer tramo: un rango hace casi todo y los otros tres esperan (4.16 s de "comunicación"). Con `cantidad` el reparto es parejo en partículas. Queda un desbalance pequeño de trabajo: el rango con las partículas sueltas hace un 15% menos de interacciones. Repartir la misma cantidad de partículas todavía no es repartir el mismo trabajo. Con `uniform` las dos particiones dan casi lo mismo (2.15 s con espacio, `results/mpi/uniform_N20000_espacio_n4_times.csv`).
