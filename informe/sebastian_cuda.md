# Barnes-Hut en CUDA

*Números de `cuda/bh_cuda.ipynb` ejecutado en Colab (Tesla T4, CUDA 13.0) el 25 de noviembre y de `results/cuda/`.*

## Diseño

`cuda/bh_cuda.cu` mantiene las partículas en la GPU en float32 (la T4 es mucho más rápida en FP32 que en FP64) y en cada paso:

1. **Ordenamiento, en la GPU**: caja con `thrust::minmax_element`, códigos de Morton con un kernel (un hilo por partícula, cálculo en double para que la celda coincida con la del árbol), `thrust::sort_by_key` y `thrust::gather` para dejar las 7 columnas y el índice original en orden Morton.
2. **Construcción, en la CPU**: se bajan posiciones, masas y códigos; el octree aplanado se arma con el mismo `c_core/` de la versión en C y se sube en float. Las copias se miden dentro de esta fase. El enunciado pide árbol aplanado, partículas ordenadas y recorrido sin recursión, pero no dice dónde se construye el árbol; *(respuesta del profesor pendiente)*.
3. **Recorrido, en la GPU**: un hilo por partícula, pila explícita en memoria local, sin recursión, mismo criterio que la versión en C (la celda propia se abre siempre).
4. **Integración, en la GPU**: kernels de medio kick y drift.

**Divergencia de warps.** Con `ORDEN_HILOS=morton` el hilo t atiende a la t-ésima partícula en orden Morton: los 32 hilos de un warp son vecinos en el espacio, abren casi las mismas celdas y siguen casi el mismo camino. Con `original` el hilo t atiende a la partícula t del archivo, el warp mezcla partículas lejanas y ejecuta la unión de sus caminos.

Sin memoria compartida ni `__syncthreads()`: cada hilo tiene su pila y escribe solo su partícula, así que no hay carreras. Cada fase se mide con `cudaEvent`.

## Verificación

`common/verify.py correr --float32 --workers 128` pasa las 5 pruebas: fuerzas con θ = 0, 1.68e-5 (umbral 1e-3); θ = 0.5, mediana 1.72e-3; trayectoria, 3.83e-7; energía en reposo, 6.62e-8; energía en movimiento, 6.14e-3 (umbral 2e-2, el mismo valor que la versión en C). `compute-sanitizer --tool memcheck`: 0 errores. `racecheck`: 0 hazards, como se esperaba sin `__shared__`.

## Rendimiento

N = 200 000, 20 pasos, 128 hilos por bloque (`results/cuda/*_times.csv`). En segundos; el total no cuenta la lectura del archivo (0.7 a 1.3 s).

| Distribución | Orden | ordenamiento | construcción | recorrido | integración | total |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | morton | 0.033 | 0.274 | 1.768 | 0.002 | 2.30 |
| uniform | original | 0.025 | 0.209 | 6.282 | 0.002 | 6.72 |
| corner | morton | 0.025 | 0.285 | 2.016 | 0.002 | 2.84 |
| corner | original | 0.024 | 0.247 | 7.261 | 0.002 | 8.07 |

Tamaño de bloque (`uniform`, `results/cuda/uniform_N200000_bloques_bench.csv`): recorrido 1.88 s con 64 hilos, 1.81 s con 128 y 1.80 s con 256.

- **Divergencia.** Con `original` el recorrido es 3.6 veces más lento en las dos distribuciones. Las partículas, el árbol y las interacciones son los mismos; solo cambia qué partículas comparten warp.
- **Fracción secuencial.** Con `morton`, la construcción en la CPU con sus copias es ≈ 12% del tiempo en `uniform` y ≈ 10% en `corner`. Aunque el recorrido fuera instantáneo, Amdahl limita la mejora a ≈ 8–10. Es lo primero que habría que llevar a la GPU.
- **Tamaño de bloque.** Casi no influye: el recorrido está limitado por las lecturas dispersas del árbol y la pila en memoria local, no por cuántos warps caben en cada multiprocesador.
- **Contra OpenMP.** En la misma prueba (`uniform`, N = 200 000, 20 pasos), OpenMP con 16 hilos en el i7-11800H tarda 7.67 s sin la lectura (recorrido 6.9 s); la T4 con `morton`, 2.30 s (recorrido 1.77 s, 3.9 veces menos). Con `original` la GPU tarda 6.72 s, casi lo mismo que la CPU: el orden de los hilos importa tanto como el hardware. El secuencial tarda 45.1 s sin lectura, así que la T4 con `morton` es ≈ 20 veces más rápida que el secuencial.
