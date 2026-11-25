# Notas de `c_omp/` (C2)

Bitácora de lo que se construyó, para estudiarlo. Las entradas nuevas van al final.

---

## C2 — OpenMP con árbol paralelo y balanceo (2026-11-25)

**Archivos creados:** `c_omp/bh_omp.c`, `c_omp/Makefile` (`make -C c_omp` → `c_omp/bh_omp`), `results/omp/{uniform,corner}_N20000_{static,dynamic}_w4_{times,workers}.csv`.

**Archivos cambiados en `c_core/`** (commit aparte en `impl/c-core`, antes de crear `impl/omp`): `aceleracion_de` ahora devuelve cuántas interacciones sumó, y `octante`, `fin_de_octante`, `reservar_nodos`, `resumir_hoja` y `resumir_interno` pasaron a `bh_core.h`. El secuencial sigue pasando las 5 pruebas.

**Resultado del verificador** (`python common/verify.py correr --cmd "./c_omp/bh_omp" --workers K`): **PASA las 5 pruebas con K = 1, 2, 4 y 8**, con los mismos valores que el secuencial (1.42e-14, 1.72e-3, 2.78e-17, 3.37e-8, 6.14e-3). Las salidas con 1 y con 8 hilos son idénticas byte a byte: cada partícula suma sus interacciones en el mismo orden, sin importar qué hilo la calcule.

**Mediciones** (N = 20 000, 50 pasos, 4 hilos, contenedor de 4 núcleos Xeon 2.1 GHz sin hyperthreading; **hay que repetirlas en el computador de la demo en C4**):

| Distribución | Schedule | Total | Segundos por hilo en fuerzas | Interacciones por hilo (millones) |
| --- | --- | --- | --- | --- |
| uniform | static | 2.67 s | 1.53 – 2.31 | 263 – 270 |
| uniform | dynamic,64 | 1.76 s | 1.48 – 1.49 | 245 – 298 |
| corner | static | 3.20 s | 1.53 – 2.84 | **241 – 353** |
| corner | dynamic,64 | 2.30 s | 2.05 – 2.05 | 290 – 323 |

- Secuencial (`c_core/bh`) con `corner`, 50 pasos: 7.30 s → speedup con 4 hilos y `dynamic` = **3.2**.
- Con `corner` y `static`, un hilo hace 1.46× más interacciones que otro: le tocaron las partículas del cúmulo, que abren más celdas. Con `dynamic,64` los hilos terminan juntos.
- Con `uniform` y `static` las interacciones son parejas pero los tiempos no: es ruido de la máquina (otro proceso le quitó un núcleo a un hilo). `dynamic` también corrige eso.
- Desglose de `corner` + `dynamic` (2.30 s): recorrido 2.06, ordenamiento 0.13 (secuencial), construcción 0.056, integración 0.004. El ordenamiento es la fracción secuencial que más pesa (≈ 6%), y por Amdahl limita el speedup a ≈ 17 aunque hubiera infinitos hilos.
- Reparto de la construcción con `corner` (`particulas_construidas`): de 142 000 a 463 000 por hilo con `static` y de 184 000 a 343 000 con `dynamic`. Está desbalanceado, pero la construcción es el 2% del tiempo, así que no importa.

**Decisiones y por qué:**
- **`schedule(runtime)` con `dynamic,64` por defecto** (`c_omp/bh_omp.c:25`, `:45` y `:172`): las partículas de un cúmulo hacen mucho más trabajo. `dynamic` reparte bloques de 64 a quien termine primero. `runtime` permite cambiarlo en la demo sin recompilar: `OMP_SCHEDULE="static" ./c_omp/bh_omp ...`.
- **`nowait` en el `for` de fuerzas** (`:172`): cada hilo anota cuándo terminó su parte. Sin `nowait` todos esperarían en la barrera del `for` y los tiempos saldrían iguales, escondiendo el desbalance. La barrera implícita del final del `parallel` (`:177`) sigue garantizando que todas las fuerzas estén listas.
- **Construcción con tasks cortadas por cantidad** (`:130`): `task if(n > N/(8·hilos))`. Un subárbol con muchas partículas se vuelve task; uno pequeño se construye ahí mismo, porque crear la task costaría más. Se corta por cantidad y no por nivel: con `corner` el 95% de las partículas cae en una sola celda de nivel 2. La perilla es `FACTOR_CORTE` (`:30`).
- **Sin candados en el árbol**: cada task trabaja en un tramo distinto de partículas (orden Morton) y escribe en su propia casilla `hijos[8k + o]`. Lo único compartido es el contador de nodos, protegido con `omp atomic capture` (`:104`).
- **Conteo previo de nodos** (`:88` y `:152`): antes de crear tasks se cuenta cuántos nodos habrá y se reserva la memoria. Si se hiciera `realloc` dentro de una task, otro hilo podría estar escribiendo en el arreglo viejo (carrera / uso de memoria liberada). El conteo es secuencial y barato (misma pasada que la construcción, sin escribir).
- **`taskwait` antes de `resumir_interno`** (`:134`): la masa de un nodo es la suma de la de sus hijos; sin esperar se leerían masas a medio calcular.
- **Caja con `reduction(min/max)`** (`:60`): cada hilo calcula su mínimo y máximo y OpenMP los combina, sin candados.
- **El ordenamiento (`qsort`) quedó secuencial** (`:202`): es la fracción secuencial que se reporta en el informe.

**Qué tendría que poder explicar el humano en la sustentación:**
- Las 3 sincronizaciones: `atomic capture` del contador de nodos (`:104`), `taskwait` (`:134`) y la barrera implícita al final del `parallel` de fuerzas (`:177`). Qué falla sin cada una: dos nodos con el mismo índice; un nodo con masa incompleta; integrar con aceleraciones a medio calcular.
- Por qué en las fuerzas no hay carrera: cada `i` escribe solo `ax[i]`, `ay[i]`, `az[i]`, y el árbol solo se lee. `mias` es privada de cada hilo y al final se guarda en la casilla `[yo]`.
- Predicción típica: "¿qué pasa de 4 a 8 hilos en una máquina de 4 núcleos?" El speedup se queda en ≈ 4 o baja un poco: los hilos extra comparten núcleo y se suma el costo de crear tasks y repartir bloques. Con hyperthreading puede subir algo (el recorrido espera memoria), pero nunca al doble.
- Por qué `static` pierde con `corner` y casi empata con `uniform` (ver la tabla).
- Modificaciones de la demo: `OMP_SCHEDULE=static|dynamic,64|guided` sin recompilar; θ con `--theta`; `FACTOR_CORTE` (`:30`) y recompilar.

**Dudas abiertas:**
- No se corrió ThreadSanitizer: con OpenMP da falsos positivos si libgomp no está compilada para él. La garantía de que no hay carreras sale del diseño (rangos disjuntos + atomic) y de que 1 y 8 hilos dan salidas idénticas.


---

## C4 — Mediciones de OpenMP en el computador de la demo (2026-11-25)

**Máquina:** portátil de Sebastián (será el de la demo): i7-11800H, 8 núcleos físicos / 16 hilos, WSL2 Ubuntu 26.04, gcc 15.2 `-O2`. Descripción completa en `results/omp/lscpu_demo.txt` (no se tocó `results/lscpu.txt` de Samuel).

**Archivos creados (`results/omp/`):**
- `uniform_N20000_bench.csv` (3 repeticiones) y `uniform_N200000_bench.csv` (3 repeticiones): 1, 2, 4, 8, 16 y 32 hilos, 20 pasos, con `common/bench.py`. Las corridas sueltas están en `bench_uniform_N*/`.
- `uniform_N20000_escalado.png/.csv` y `uniform_N200000_escalado.png/.csv`: tiempo, speedup, eficiencia y Karp-Flatt con `common/plot.py`. T1 = secuencial de C1 (`secuencial_uniform_N*_times.csv`).
- `{uniform,corner}_N20000_{static,dynamic}_w{4,8}_100p_{times,workers}.csv`: balance de carga, 100 pasos.
- `uniform_N*_mntc_*`: la primera corrida, leyendo desde `/mnt/c`. Se dejó como evidencia del hallazgo de abajo.

**Hallazgo importante: correr desde el disco de WSL, no desde `/mnt/c`.** Leer `uniform_N200000` desde `/mnt/c/...` (la carpeta de OneDrive vista desde WSL) tarda 14.3 s; desde `~/` tarda 0.41 s (35× menos). Esa lectura y la escritura de `acc0.txt` son secuenciales y entran en `total`, así que en la primera corrida el speedup se quedaba en 2.2× con 16 hilos y Karp-Flatt daba ≈ 0.43 constante: Amdahl puro por E/S. Las mediciones buenas se corrieron en una copia del repo en `~/bh-medicion`. **En la demo hay que correr desde `~/`** (copiar el repo o al menos `data/` y las salidas a `/tmp`).

**Escalado (total, promedio de 3, T1 secuencial: 3.69 s con N = 20 000, 45.46 s con N = 200 000):**

| Hilos | N=20k: s | Speedup | Eficiencia | N=200k: s | Speedup | Eficiencia |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.25 | 1.13 | 1.13 | 43.99 | 1.03 | 1.03 |
| 2 | 1.90 | 1.94 | 0.97 | 25.41 | 1.79 | 0.89 |
| 4 | 1.15 | 3.21 | 0.80 | 15.54 | 2.93 | 0.73 |
| 8 | 0.68 | 5.42 | 0.68 | 10.10 | 4.50 | 0.56 |
| 16 | 0.55 | 6.67 | 0.42 | 8.08 | 5.63 | 0.35 |
| 32 | 0.60 | 6.18 | 0.19 | 8.28 | 5.49 | 0.17 |

Lectura: hasta 8 hilos (núcleos físicos) escala bien. De 8 a 16 se gana poco (hyperthreading: dos hilos comparten las unidades de un núcleo). Con 32 hay más hilos que CPUs lógicas y empeora un poco (sobresuscripción). Con 1 hilo da un poco más rápido que el secuencial: es ruido del turbo del portátil, no un efecto real. Con N = 200 000 escala algo peor porque el árbol (69 343 nodos) ya no cabe en caché y el recorrido queda limitado por la memoria.

**Balance de carga (N = 20 000, 100 pasos; `uniform` dt = 0.001, `corner` dt = 2.5e-4):**

| Caso | static: total | static: interacciones máx/mín | dynamic,64: total | dynamic: máx/mín |
| --- | --- | --- | --- | --- |
| uniform, 4 hilos | 5.52 s | 1.02 | 4.94 s | 1.09 |
| uniform, 8 hilos | 4.10 s | 1.05 | 3.47 s | 1.17 |
| corner, 4 hilos | 8.63 s | 2.02 | 7.43 s | 1.08 |
| corner, 8 hilos | 6.73 s | 2.76 | 4.64 s | 1.09 |

Lectura: con `corner` y `static`, un hilo hace 2.8× más interacciones que otro (8 hilos), y sus tiempos van de 1.5 s a 5.6 s: todos esperan al más cargado. `dynamic,64` reparte en trozos de 64 partículas a quien termine primero y deja los tiempos de los hilos iguales (4.17 s todos): 31 % menos de tiempo total. Con `uniform` la diferencia es menor porque todas las partículas cuestan casi lo mismo.

**Qué tendría que poder explicar el humano en la sustentación:**
- Speedup, eficiencia y Karp-Flatt: qué significa que Karp-Flatt quede casi constante (parte secuencial, Amdahl) o que crezca (overhead).
- Por qué de 8 a 16 hilos casi no mejora (8 núcleos físicos) y por qué 32 empeora.
- Por qué `dynamic` gana con `corner` (columna de interacciones de `workers.csv`).
- El hallazgo de `/mnt/c`: la E/S secuencial también es parte de Amdahl.

**Dudas abiertas:** ninguna.
