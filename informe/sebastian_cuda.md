# Barnes-Hut en CUDA

**Diseño.** `cuda/bh_cuda.cu` mantiene las partículas en la GPU en float32 (la T4 es mucho más
rápida en FP32) y en cada paso:

1. **Orden en la GPU**: caja con `thrust::minmax_element`, códigos de Morton con un kernel (en
   double, para que la celda coincida con la del árbol), `thrust::sort_by_key` y `thrust::gather`.
2. **Construcción en la CPU**: el octree aplanado se arma con el mismo `c_core/` de la versión en C
   y se sube en float; las copias se miden en esta fase. El profesor confirmó que es aceptable: el
   enunciado no exige construir el octree en la GPU.
3. **Recorrido en la GPU, sin recursión**: un hilo por partícula con pila explícita en memoria local.
4. **Integración en la GPU**: kernels de medio kick y drift.

Sin memoria compartida ni `__syncthreads()`: cada hilo tiene su pila y escribe solo su partícula,
así que no hay carreras (`racecheck`: 0 hazards).

**Divergencia de warps.** Con `ORDEN_HILOS=morton` el hilo t atiende a la t-ésima partícula en orden
Morton: los 32 hilos de un warp son vecinos, abren casi las mismas celdas y siguen casi el mismo
camino. Con `original` el warp mezcla partículas lejanas y ejecuta la unión de sus caminos.

**Desglose por fase** (Colab, Tesla T4, N = 200 000, 20 pasos, 128 hilos por bloque,
`results/cuda/*_times.csv`; segundos, sin la lectura del archivo):

| Distribución | Orden | ordenamiento | construcción | recorrido | integración | total |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | morton | 0.033 | 0.274 | 1.768 | 0.002 | 2.30 |
| uniform | original | 0.025 | 0.209 | 6.282 | 0.002 | 6.72 |
| corner | morton | 0.025 | 0.285 | 2.016 | 0.002 | 2.84 |
| corner | original | 0.024 | 0.247 | 7.261 | 0.002 | 8.07 |

- **Divergencia**: con `original` el recorrido es 3.6 veces más lento en las dos distribuciones,
  con las mismas partículas, árbol e interacciones; solo cambia qué partículas comparten warp.
- **Fracción secuencial**: con `morton`, la construcción en la CPU con sus copias es ≈ 12% del
  tiempo en `uniform` y ≈ 10% en `corner`; por Amdahl limita la mejora a ≈ 8–10.
- **Tamaño de bloque** casi no influye (recorrido 1.88, 1.81 y 1.80 s con 64, 128 y 256 hilos): lo
  limitan las lecturas dispersas del árbol, no la ocupación.
- **Contra OpenMP** (`uniform`, N = 200 000): 16 hilos del i7-11800H tardan 7.67 s sin lectura; la
  T4 con `morton`, 2.30 s. Con `original` la GPU tarda 6.72 s: el orden de los hilos importa casi
  tanto como el hardware.
