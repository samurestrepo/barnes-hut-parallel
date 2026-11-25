# Barnes-Hut con OpenMP (C)

*Borrador sin pulir. Los números salen de `results/omp/`; la fuente va entre paréntesis. Las mediciones de escalado (1, 2, 4, 8 hilos y sobresuscripción) se completan en C4, en el computador de la demo.*

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

## Rendimiento (preliminar)

N = 20 000, 50 pasos, 4 hilos, máquina de 4 núcleos sin hyperthreading (`results/omp/*_N20000_*_w4_*.csv`). Secuencial con `corner`: 7.30 s.

| Distribución | Schedule | Total | Interacciones por hilo (M) | Segundos por hilo |
| --- | --- | --- | --- | --- |
| uniform | static | 2.67 s | 263 – 270 | 1.53 – 2.31 |
| uniform | dynamic,64 | 1.76 s | 245 – 298 | 1.48 – 1.49 |
| corner | static | 3.20 s | 241 – 353 | 1.53 – 2.84 |
| corner | dynamic,64 | 2.30 s | 290 – 323 | 2.05 – 2.05 |

- **Balanceo.** Con `corner` y `static`, un hilo hace 1.46× más interacciones que otro y el total sube 39%. Con `dynamic` los hilos terminan juntos. Speedup con 4 hilos y `dynamic`: 3.2.
- **Fracción secuencial.** Con `corner` + `dynamic` el desglose es: recorrido 2.06 s, ordenamiento 0.13 s (secuencial), construcción 0.056 s, integración 0.004 s. El ≈ 6% del `qsort` limita el speedup a ≈ 17 por Amdahl.
- **Construcción desbalanceada pero barata**: con `corner` los hilos pusieron de 142 000 a 463 000 partículas en hojas (`static`), pero la construcción es el 2% del tiempo.
- *(C4)* Escalado con 1, 2, 4, 8 hilos y hasta 2× los núcleos lógicos con `common/bench.py`; Karp-Flatt; por qué se aplana (núcleos físicos frente a hilos virtuales, ancho de banda de memoria al leer el árbol, fracción secuencial).
