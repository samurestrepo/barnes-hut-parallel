# Verificación de correctitud

`common/verify.py correr --cmd "<versión>"` corre la versión con los parámetros del contrato y la
compara contra una suma directa O(N²) en NumPy, con N = 2000. Cinco pruebas:

| Prueba | Qué detecta | Umbral (float64 / float32) | S2 y S3 | S4 (4 rangos) | C y OpenMP | CUDA |
| --- | --- | --- | --- | --- | --- | --- |
| Fuerzas, θ = 0 | árbol o fórmula mal | 1e-6 / 1e-3 | 1.49e-14 | 1.49e-14 | 1.42e-14 | 1.68e-5 |
| Fuerzas, θ = 0.5 (mediana) | criterio de apertura | 1e-2 | 3.25e-3 | 3.24e-3 | 1.72e-3 | 1.72e-3 |
| Trayectoria, 10 pasos | integrador u orden de salida | 1e-6 / 1e-3 | 2.78e-17 | 2.78e-17 | 2.78e-17 | 3.83e-7 |
| Energía en reposo, 100 pasos | integrador | 1e-5 / 1e-3 | 3.37e-8 | 3.37e-8 | 3.37e-8 | 6.62e-8 |
| Energía en movimiento, 100 pasos | todo junto, con cúmulos | 2e-2 | 8.87e-3 | 8.98e-3 | 6.14e-3 | 6.14e-3 |

Todas las versiones pasan las cinco con cualquier número de trabajadores: multiprocessing y OpenMP
dan salidas idénticas bit a bit a su versión secuencial, y MPI también con θ = 0. En CUDA,
`compute-sanitizer --tool memcheck` da 0 errores y `racecheck` 0 *hazards* (no usa memoria
compartida). *(Los valores de S4 se midieron en Windows con Microsoft MPI; confirmar con
`results/demo_pc/`.)*

**El umbral de la energía en movimiento.** Se empezó con 1e-3 y ninguna versión correcta lo
pasaba. Con θ = 0 la misma prueba da 4.0e-4, igual que la suma directa, y a tiempo físico fijo la
deriva no baja al reducir dt: no es el integrador, es el error de aproximar celdas por su centro de
masa mientras los cúmulos colapsan. Se fijó en 2e-2, que sigue detectando un leapfrog sin el
segundo medio kick (2.0e-1). Los errores del integrador los detectan además la trayectoria y la
energía en reposo.

**Por qué el mismo θ da distinta precisión en Python y en C.** Con θ = 0.5 sobre el mismo archivo,
la mediana del error es 3.2e-3 en Python y 1.7e-3 en C (máximo 4.4e-2 y 2.1e-2). Las dos son
Barnes-Hut correctos, con una diferencia de diseño en el recorrido: Python aplica el criterio θ
también a las hojas, así que una hoja lejana cuenta como una sola masa (`py_core/bh_core.py`,
`aceleracion_de`); C suma siempre las partículas de una hoja una por una
(`c_core/bh_fuerzas.c`, `aceleracion_de`). C es más preciso y hace más interacciones por partícula.
Además, el cubo raíz de Python empieza en la esquina mínima de las posiciones y el de C está
centrado, así que los árboles no son idénticos. Por eso los tiempos de Python y de C no se comparan
como si hicieran el mismo trabajo, y dentro de cada lenguaje sí.
