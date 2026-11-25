# Barnes-Hut con OpenMP (C)

*Los números salen de `results/omp/` y se midieron en el computador de la demo: i7-11800H, 8 núcleos físicos / 16 hilos, WSL2, gcc 15.2 (`results/omp/lscpu_demo.txt`).*

## Diseño

El núcleo secuencial (`c_core/`) arma en cada paso la caja de las partículas, sus códigos de Morton de 21 bits por eje, las ordena con `qsort` por (código, índice) y construye un octree **aplanado** en arreglos (hijos, inicio y cuenta de partículas, masa, centro de masa, centro y mitad del lado), con hojas de hasta 8 partículas. Al estar ordenadas por Morton, las partículas de cada celda forman un tramo contiguo: el árbol se construye partiendo rangos, sin insertar una por una. El recorrido usa una pila explícita y abre siempre la celda que contiene a la partícula. El integrador es leapfrog kick-drift-kick. Las mismas estructuras las usa la versión CUDA.

`c_omp/bh_omp.c` paraleliza sobre ese núcleo:

- **Fuerzas** (≈ 90% del tiempo): `#pragma omp for schedule(runtime)`, con `dynamic,64` por defecto. Las partículas de un cúmulo denso abren muchas más celdas que las aisladas; con `dynamic` los bloques de 64 se reparten a quien termine primero. `runtime` permite cambiarlo con `OMP_SCHEDULE` sin recompilar.
- **Construcción del árbol sin candados**: tareas recursivas `#pragma omp task if(n > N/(8·hilos))`, cortadas por **cantidad de partículas** y no por nivel. Un plan anterior con 64 subárboles fijos de nivel 2 dejaba el 95% de `corner` en un solo subárbol. Cada tarea trabaja en su tramo y escribe en su casilla de `hijos`; lo único compartido es el contador de nodos (`omp atomic capture`). La memoria se reserva antes con un conteo de nodos, para no hacer `realloc` entre hilos. `taskwait` asegura que la masa de un nodo se calcule después que la de sus hijos.
- **Caja** con `reduction(min/max)`, **códigos Morton** e **integración** con `parallel for`.
- **Secuencial**: el `qsort` y el conteo de nodos.

Cada hilo anota sus segundos, sus interacciones y las partículas que puso en hojas en `PREFIJO_workers.csv`.

## Verificación

Pasa las 5 pruebas de `common/verify.py` con 1, 2, 4 y 8 hilos, con los mismos valores que el secuencial: fuerzas con θ = 0, 1.42e-14; con θ = 0.5, mediana 1.72e-3; trayectoria de 10 pasos, 2.78e-17; energía en reposo, 3.37e-8; energía en movimiento, 6.14e-3 (umbral 2e-2; Python da 8.9e-3 por la misma aproximación de monopolo). Las salidas con 1 y 8 hilos son idénticas byte a byte: cada partícula suma en el mismo orden sin importar qué hilo la calcula, lo que también descarta carreras en las fuerzas.

## Rendimiento

**Escalado** (`uniform`, 20 pasos, promedio de 3, `results/omp/uniform_N*_bench.csv` y `*_escalado.png`). T1 es el secuencial de C1: 3.69 s con N = 20 000 y 45.46 s con N = 200 000.

| Hilos | N = 20 000 | Speedup | Eficiencia | Karp-Flatt | N = 200 000 | Speedup | Eficiencia | Karp-Flatt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1.90 s | 1.94 | 0.97 | 0.03 | 25.41 s | 1.79 | 0.89 | 0.12 |
| 4 | 1.15 s | 3.21 | 0.80 | 0.08 | 15.54 s | 2.93 | 0.73 | 0.12 |
| 8 | 0.68 s | 5.42 | 0.68 | 0.07 | 10.10 s | 4.50 | 0.56 | 0.11 |
| 16 | 0.55 s | 6.67 | 0.42 | 0.09 | 8.08 s | 5.63 | 0.35 | 0.12 |
| 32 | 0.60 s | 6.18 | 0.19 | 0.14 | 8.28 s | 5.49 | 0.17 | 0.16 |

- **Hasta 8 hilos escala bien**: son los 8 núcleos físicos.
- **De 8 a 16 gana poco** (5.42 → 6.67): los hilos 9 a 16 son hyperthreading, dos hilos comparten las unidades de un núcleo. Ayuda algo porque el recorrido pasa mucho tiempo esperando memoria.
- **Con 32 empeora** (sobresuscripción): más hilos que CPUs lógicas, el sistema los turna y se pierde en cambios de contexto. Karp-Flatt sube a 0.14–0.16, señal de sobrecosto y no de parte secuencial.
- **Con N = 200 000 escala peor**: el árbol tiene 69 343 nodos y ya no cabe en caché, así que los hilos compiten por el ancho de banda de memoria. Karp-Flatt se queda en ≈ 0.12 de 2 a 16 hilos.
- **Fracción secuencial medida**: con 16 hilos y N = 200 000, el `qsort` (0.45 s) y la lectura (0.41 s) son ≈ 11% del total, cerca del 0.12 de Karp-Flatt.
- **La E/S también es Amdahl**: leyendo desde `/mnt/c` (OneDrive visto desde WSL) la lectura de 200 000 partículas tarda 14.3 s en lugar de 0.41 s, y el speedup con 16 hilos cae a 2.2 (`results/omp/*_mntc_*`). Las mediciones y la demo corren desde `~/`.

**Balanceo** (N = 20 000, 100 pasos, `results/omp/*_100p_*.csv`):

| Distribución | Hilos | static | dynamic,64 | Interacciones máx/mín con static |
| --- | --- | --- | --- | --- |
| uniform | 4 | 5.52 s | 4.94 s | 1.02 |
| uniform | 8 | 4.10 s | 3.47 s | 1.05 |
| corner | 4 | 8.63 s | 7.43 s | 2.02 |
| corner | 8 | 6.73 s | 4.64 s | 2.76 |

Con `corner` y `static`, al hilo al que le toca el cúmulo le tocan las partículas que más celdas abren: hace 2.8 veces más interacciones que el menos cargado, y los hilos terminan entre 1.5 y 5.6 s. Con `dynamic,64` la razón baja a 1.09, todos terminan a los 4.17 s y el total baja 31%. Con `uniform` las partículas cuestan casi lo mismo y la diferencia es menor.

**Comparación con CUDA** (misma prueba, `uniform`, N = 200 000, 20 pasos): OpenMP con 16 hilos, 7.67 s sin la lectura; la T4 con `morton`, 2.30 s (ver `informe/sebastian_cuda.md`).
