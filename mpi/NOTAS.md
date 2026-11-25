# Notas de `mpi/` (S4)

Bitácora para estudiar en otro chat. Las entradas nuevas van al final.

---

## Plan (escrito antes de codificar, 2026-11-24)

**Qué se paraleliza:**
- Los códigos de Morton: cada rango calcula los de sus partículas con la caja global (`Allreduce` de mínimos y máximos).
- El árbol: cada rango construye el árbol de su trozo (≈ N/P partículas contiguas en orden Morton).
- La poda: cada rango recorre su árbol contra la caja de cada rango remoto y arma lo que le tiene que mandar (celdas aceptadas como (cm, masa) y partículas de las hojas cercanas).
- Las fuerzas: cada rango arma un árbol con sus partículas + lo recibido y calcula solo las aceleraciones de las suyas (`aceleraciones_lista` de `py_core`).
- La integración (kick, drift, kick) de las partículas propias.

**Qué queda secuencial:** el rango 0 recibe códigos y partículas (`Gatherv`), las ordena (`argsort`), corta en P trozos y las reparte (`Scatterv`), en cada paso. Mientras tanto los demás esperan. También la lectura y la escritura de archivos (solo el rango 0).

**Speedup teórico (Amdahl) con la fracción secuencial medida en S2** (`results/py_core/resumen_uniform_N20000.csv`, uniform N = 20 000, 20 pasos):
- Ordenamiento en S2 = 2.9% de la simulación (incluye los códigos de Morton, que en MPI se paralelizan, así que es una cota alta). La lectura es 2.8% del total. Fracción secuencial f ≈ 0.029 sin contar la lectura, ≈ 0.057 contándola.
- S(P) = 1 / (f + (1 − f)/P). Con f = 0.057: S(2) = 1.89, S(4) = 3.42, S(8) = 5.72. Con f = 0.029: S(2) = 1.94, S(4) = 3.68, S(8) = 6.63.
- Lo que Amdahl no cuenta y bajará el speedup real: (1) la comunicación (Gatherv/Scatterv de todas las partículas en cada paso, Alltoallv de las ramas podadas), (2) el trabajo extra: cada rango arma un segundo árbol con lo recibido, y (3) **esta máquina tiene 2 núcleos físicos y 4 lógicos** (i5-6300U): con 4 rangos dos comparten núcleo por hyper-threading y con 8 hay sobresuscripción. Espero S(2) ≈ 1.6–1.8, S(4) ≈ 2–2.5 y S(8) ≤ S(4).

**Cómo se mide:**
- Cada rango mide sus fases con `time.perf_counter()`; el rango 0 junta las de todos (`Gather`) y escribe en `times.csv` la del **rango más lento** en cada fase. `total` es el reloj de pared del rango 0 (lectura + simulación, sin el calentamiento de Numba).
- `comunicacion` = todas las colectivas. Ojo: incluye la **espera** por el rango más lento (un rango que termina antes espera dentro de la colectiva siguiente), así que un mal balanceo aparece como comunicación.
- `ordenamiento` = códigos de Morton + `argsort` del rango 0; `construccion` = árbol local + masas + poda + árbol combinado.
- `workers.csv`: por rango, segundos de recorrido, interacciones, partículas promedio por paso, bytes recibidos y bytes que habría recibido con los árboles completos de los demás.
- Speedup contra S2 secuencial (0.161 s por paso) con `uniform` N = 20 000, 20 pasos, con 1, 2, 4 y 8 rangos.
- Bytes: recibidos en el `Alltoallv` frente a Σ (nodos + partículas) × 32 bytes de los árboles de los otros rangos.

**Entorno:** Windows con Microsoft MPI 10.1 y mpi4py 4.1.2. Se lanza con `mpiexec -n K` (no hay `mpirun`). MS-MPI deja correr más rangos que núcleos sin `--oversubscribe`.

---

## S4 — MPI con Morton y ramas remotas (2026-11-24)

**Versión:** la completa del diseño (ramas podadas con la distancia mínima a la caja remota). **No** se usó la simplificación de "Si te atrasas".

**Archivos creados:**
- `mpi/bh_mpi.py`: la versión MPI. Importa el árbol, el recorrido y el leapfrog de `py_core/bh_core.py` (S2).
- `results/mpi/uniform_N20000_cantidad_n{1,2,4,8}_r{1,2}_{times,workers}.csv` y `resumen_uniform_N20000.csv`.
- `results/mpi/corner_N20000_{cantidad,espacio}_n4_*` y `uniform_N20000_espacio_n4_*`: comparación de particiones.
- `results/mpi/sin_limite_giro/`: las mismas corridas antes de fijar `MPICH_PROGRESS_SPIN_LIMIT` (evidencia del problema con 8 rangos, ver abajo).

**Archivos cambiados:** `CONTRATO_DUDAS.md` (se usa `mpiexec` de MS-MPI en vez de `mpirun`).

**Resultado del verificador** (`python common/verify.py correr --cmd "mpiexec -n K python mpi/bh_mpi.py --particion cantidad"`):

| Prueba | -n 1 | -n 2 | -n 4 |
| --- | --- | --- | --- |
| fuerzas_directa | PASA 1.49e-14 | PASA 1.49e-14 | PASA 1.49e-14 |
| fuerzas_bh | PASA 3.25e-3 | PASA 3.25e-3 | PASA 3.24e-3 |
| trayectoria | PASA 2.78e-17 | PASA 2.78e-17 | PASA 2.78e-17 |
| energia_reposo | PASA 3.37e-8 | PASA 3.37e-8 | PASA 3.37e-8 |
| energia_movimiento | FALLA 8.87e-3 | FALLA 8.90e-3 | FALLA 8.98e-3 |

`energia_movimiento` falla por el umbral provisional 1e-3, igual que S2 (propuesta 2e-2 en `CONTRATO_DUDAS.md`). Con θ = 0 las salidas son idénticas bit a bit a S2 con 1, 2 y 4 rangos: nada se acepta, viajan todas las partículas y el árbol combinado es el mismo árbol global. Con θ = 0.5 cambian un poco (mediana 5e-4 de diferencia relativa en acc0 contra S2) porque las celdas aceptadas llegan como una partícula y el árbol combinado vuelve a aproximarlas; el error contra la suma directa queda igual (3.24e-3 a 3.25e-3).

**Bytes recibidos frente al árbol completo** (`uniform` N = 20 000, 20 pasos = 21 cálculos de fuerza, suma de todos los rangos; filas `bytes_*` de times.csv):

| Rangos | Recibidos | Árbol completo de los demás | Proporción |
| --- | --- | --- | --- |
| 2 | 5 777 664 B (5.8 MB) | 17 622 624 B (17.6 MB) | 32.8% |
| 4 | 16 332 160 B (16.3 MB) | 52 867 200 B (52.9 MB) | 30.9% |
| 8 | 33 449 600 B (33.4 MB) | 123 440 800 B (123.4 MB) | 27.1% |

"Árbol completo" = lo que llegaría si cada rango mandara todos sus nodos como (cm, masa) y todas sus partículas: Σ (nodos + partículas) × 32 bytes de los otros rangos (`bytes_arbol_completo`, `mpi/bh_mpi.py:242`). Siempre se recibe menos de un tercio.

**Tiempos** (`uniform` N = 20 000, θ = 0.5, 20 pasos, `--particion cantidad`, promedio de 2 corridas; S2 = 3.315 s; `results/mpi/resumen_uniform_N20000.csv`):

| Rangos | total (s) | por paso (s) | speedup | eficiencia | comunicación (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.745 | 0.180 | 0.89 | 0.89 | 0.023 |
| 2 | 2.506 | 0.121 | 1.32 | 0.66 | 0.135 |
| 4 | 2.085 | 0.099 | 1.59 | 0.40 | 0.299 |
| 8 | 2.273 | 0.103 | 1.46 | 0.18 | 0.936 |

Contra Amdahl (Plan): se esperaba S(2) = 1.89 y S(4) = 3.42 con f = 0.057; se midió 1.32 y 1.59. La diferencia la explica sobre todo el hardware (2 núcleos físicos: de 2 a 4 rangos solo se gana hyper-threading) y luego la comunicación (de 0.02 s a 0.30 s con 4 rangos, incluida la espera en colectivas) y el árbol combinado (construcción 0.20 → 0.25 s). Con 1 rango es 13% más lento que S2: arma dos árboles por paso (el local y el combinado) y pasa por las colectivas aunque no haya con quién hablar.

**Algo raro que se encontró y se corrigió (8 rangos):** en la primera medición, 8 rangos tardaban 9.6 s (speedup 0.35) con 8.0 s de "comunicación", aunque cada rango calculaba solo ≈ 1 s (`results/mpi/sin_limite_giro/uniform_N20000_cantidad_n8_r*`). Hipótesis: Microsoft MPI espera **girando** (busy-wait) dentro de las colectivas; con 8 procesos en 4 hilos de hardware, los que esperan le quitan CPU a los que calculan y los rangos terminan turnándose (≈ 8 × 1 s). Se comprobó: con `MPICH_PROGRESS_SPIN_LIMIT=16` (5 pasos) la comunicación con 8 rangos bajó de 2.2 s a 0.3 s, y con 4 rangos el total bajó de 0.84 a 0.61 s; con 2 rangos no cambia. Se dejó fijado arriba del archivo (`mpi/bh_mpi.py:29`, perilla). multiprocessing (S3) no tiene este problema porque sus esperas bloquean en una tubería, no giran.

**Espacio vs cantidad con `corner`** (N = 20 000, dt = 2.5e-4, 4 rangos, 20 pasos, `results/mpi/corner_N20000_*_n4_workers.csv`):

| Partición | total (s) | partículas por rango | recorrido por rango (s) | interacciones por rango (millones) | comunicación (s) |
| --- | --- | --- | --- | --- | --- |
| cantidad | 2.405 | 5000 / 5000 / 5000 / 5000 | 1.79 / 1.79 / 1.74 / 1.60 | 62.5 / 63.3 / 61.6 / 53.2 | 0.30 |
| espacio | 4.748 | 19 254 / 241 / 252 / 253 | 4.16 / 0.03 / 0.03 / 0.03 | 241.8 / 1.0 / 1.0 / 1.0 | 4.16 |

Con `espacio`, el rincón denso (95% de las partículas en [0, 0.05]³) cae entero en el primer tramo del espacio de códigos: el rango 0 hace casi todo y los otros 3 esperan (su espera aparece como 4.16 s de comunicación). Con `cantidad` cada rango tiene N/P partículas; queda un desbalance pequeño de trabajo (el rango 3 tiene las partículas sueltas de afuera y hace 15% menos interacciones), que es el "cantidad todavía no es trabajo" del diseño. Con `uniform`, `espacio` reparte casi parejo (4909–5049 partículas) y tarda 2.15 s, como `cantidad`.

**Decisiones y por qué:**
- **Cada partícula viaja como 8 floats** (x y z vx vy vz m índice, `COLUMNAS`, `:51`): un solo búfer por colectiva. El índice original viaja con ella para devolver todo al orden de la entrada al final (`:304`). Los códigos van aparte (int64): un código de 63 bits no cabe exacto en un float64.
- **El árbol local usa la caja GLOBAL** (`fuerzas_locales`, `:250`): así sus celdas son celdas del octree global y los códigos que mandó el rango 0 sirven sin recalcular.
- **Poda con la distancia mínima** (`acepta_para_caja`, `:87`): si la celda cumple θ vista desde el punto más cercano de la caja remota, la cumple para cualquier partícula de esa caja (todas están más lejos). Además, si la celda **toca** la caja remota se abre siempre (`:90`): podría haber una partícula remota dentro, y el contrato dice que esa celda se abre.
- **Lo recibido se mete en un árbol junto con las partículas propias** (`aceleraciones_con_recibido`, `:271`) y se recorre con `aceleraciones_lista` de S2 solo para las propias (`:281`). Alternativa descartada: suma directa contra todo lo recibido; con 4 rangos son miles de puntos por partícula, mucho más caro que el recorrido.
- **`--particion espacio`** = cortar el espacio de códigos [0, 2^63) en P tramos iguales (`:147`): regiones de igual volumen, el reparto ingenuo. **`cantidad`** = cortar el arreglo ordenado en P tramos de N/P (`:145`).
- **Solo el rango 0 lee** (`leer_en_cero`); los demás empiezan vacíos y reciben su trozo en el primer reparto, que es el mismo código de todos los pasos.
- **`times.csv` lleva, por fase, el tiempo del rango más lento** (`tiempos_del_mas_lento`); `total` es el reloj de pared del rango 0. `workers.csv` trae por rango los segundos de recorrido, interacciones, partículas por paso y bytes.
- **Calentamiento**: el 0 compila y escribe la caché de Numba, los demás esperan en una barrera y la cargan, y todos salen juntos por otra barrera (`calentar`, `:330`; `:345`). Sin la última barrera, el 0 medía como comunicación ≈ 0.5 s de espera mientras los demás cargaban Numba.

**Qué tendría que poder explicar el humano en la sustentación:**
- **El orden de las colectivas** (marcadas `# SINCRONIZACIÓN:` y numeradas 1–11 por paso): Allreduce ×2 (`:134`), Gather + Gatherv ×2, Scatter + Scatterv ×2 (`:179`), Allgather (`:210`), Alltoall + Alltoallv (`:237`). Todos los rangos las llaman todas y en el mismo orden, también el que no tiene partículas (con búferes vacíos). El trabajo extra del rango 0 (ordenar, `:172`) va **entre** dos colectivas, no dentro de un `if` que se salte alguna. Si un rango se saltara una colectiva o las llamara en otro orden, todos se quedarían esperando: deadlock (bug típico 2 de la hoja).
- **Por qué no hay deadlock de Send/Send** (bug típico 1): no hay envíos punto a punto; el intercambio de ramas es un `Alltoallv` (`:237`) y MPI organiza los envíos.
- **Por qué hay un `Alltoall` antes del `Alltoallv`**: el que recibe tiene que saber cuánto le llega de cada uno para reservar el búfer.
- **La fracción secuencial**: el rango 0 ordena mientras los demás esperan dentro del `Scatter` (`:179`). Con N = 20 000 es poco (ordenamiento 0.03–0.04 s de 2.1 s), pero crece como N log N y no se reparte.
- **La poda en un dibujo**: caja remota, celda lejana → viaja su (cm, masa); celda cercana → se abre; hoja cercana → viajan sus partículas.
- **Lo que muestra `workers.csv` con `corner`**: espacio → un rango con 19 254 partículas y 4.2 s, tres casi ociosos; cantidad → 5000 cada uno y 1.6–1.8 s.
- Perillas: `--particion` (CLI), `--theta` (CLI), `MPICH_PROGRESS_SPIN_LIMIT` (`:29`).

**Dudas abiertas:**
- Umbral de `energia_movimiento` (igual que S2).
- En WSL2/OpenMPI (el entorno del diseño) no se pudo probar; ahí se lanza con `mpirun --oversubscribe -n K` y la perilla de giro no aplica.
- El rango 0 ordena con `argsort` de NumPy: con N = 200 000 se verá cuánto pesa (S5).
