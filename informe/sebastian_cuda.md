# Barnes-Hut en CUDA

*Borrador sin pulir. Los números se completan con el notebook ejecutado en Colab (`cuda/bh_cuda.ipynb`, GPU T4) y `results/cuda/`.*

## Diseño

`cuda/bh_cuda.cu` mantiene las partículas en la GPU en float32 (la T4 es mucho más rápida en FP32 que en FP64) y en cada paso:

1. **Ordenamiento, en la GPU**: caja con `thrust::minmax_element`, códigos de Morton con un kernel (un hilo por partícula, cálculo en double para que la celda coincida con la del árbol), `thrust::sort_by_key` y `thrust::gather` para dejar las 7 columnas y el índice original en orden Morton.
2. **Construcción, en la CPU**: se bajan posiciones, masas y códigos; el octree aplanado se arma con el mismo `c_core/` de la versión en C y se sube en float. Las copias se miden dentro de esta fase. El enunciado pide árbol aplanado, partículas ordenadas y recorrido sin recursión, pero no dice dónde se construye el árbol; *(respuesta del profesor pendiente)*.
3. **Recorrido, en la GPU**: un hilo por partícula, pila explícita en memoria local, sin recursión, mismo criterio que la versión en C (la celda propia se abre siempre).
4. **Integración, en la GPU**: kernels de medio kick y drift.

**Divergencia de warps.** Con `ORDEN_HILOS=morton` el hilo t atiende a la t-ésima partícula en orden Morton: los 32 hilos de un warp son vecinos en el espacio, abren casi las mismas celdas y siguen casi el mismo camino. Con `original` el hilo t atiende a la partícula t del archivo, el warp mezcla partículas lejanas y ejecuta la unión de sus caminos.

Sin memoria compartida ni `__syncthreads()`: cada hilo tiene su pila y escribe solo su partícula, así que no hay carreras. Cada fase se mide con `cudaEvent`.

## Verificación

*(Celda 3 del notebook)* `common/verify.py correr --float32 --workers 128`: fuerzas con θ = 0, ___; θ = 0.5, ___; trayectoria, ___; energía en reposo, ___; energía en movimiento, ___. `compute-sanitizer --tool memcheck`: ___ errores.

## Rendimiento

*(Celdas 5 y 6)* N = 200 000, 20 pasos, 128 hilos por bloque:

| Distribución | Orden | ordenamiento | construcción | recorrido | integración | total |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | morton | | | | | |
| uniform | original | | | | | |
| corner | morton | | | | | |
| corner | original | | | | | |

Tamaño de bloque (64, 128, 256): ___.

Análisis a completar: qué fracción es la construcción en la CPU con las copias (la parte secuencial, que limita el speedup como en Amdahl), cuánto empeora el recorrido con `original` (divergencia) y comparación con OpenMP en la misma N.
