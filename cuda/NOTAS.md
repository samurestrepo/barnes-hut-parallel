# Notas de `cuda/` (C3)

Bitácora de lo que se construyó, para estudiarlo. Las entradas nuevas van al final.

---

## C3 — CUDA con árbol aplanado (2026-11-25)

**Archivos creados:**
- `cuda/bh_cuda.cu`: kernels y bucle de la simulación en la GPU.
- `cuda/Makefile`: compila `c_core/` con gcc (como C) y enlaza con nvcc (`-arch=sm_75`, la T4).
- `cuda/bh_cuda.ipynb`: notebook de Colab: GPU, compilación, verificador, compute-sanitizer, desglose por fase (`morton` vs `original`) y tamaños de bloque. **Se sube ejecutado** (lo pide el enunciado).

**Estado:** compila sin avisos con nvcc 12.0 para `sm_75`. **Falta correrlo en Colab**: el verificador, compute-sanitizer y las mediciones salen del notebook. Cuando corra, completar la tabla de abajo.

| Prueba (`--float32 --workers 128`) | Resultado | Valor | Umbral |
| --- | --- | --- | --- |
| fuerzas_directa | pendiente | | 1e-3 |
| fuerzas_bh | pendiente | | 1e-2 |
| trayectoria | pendiente | | 1e-3 |
| energia_reposo | pendiente | | 1e-3 |
| energia_movimiento | pendiente | | 2e-2 |

**Cómo es un paso** (`calcular_fuerzas`, `cuda/bh_cuda.cu:317`):
1. **ordenamiento** (`ordenar_gpu`, `:242`): caja con `thrust::minmax_element`, códigos Morton con `kernel_codigos` (un hilo por partícula), `thrust::sort_by_key` (`:248`) y `thrust::gather` para mover las 7 columnas y el `id` a ese orden (sin copiar de vuelta: se intercambian punteros).
2. **construccion** (`construir_en_cpu`, `:299`): baja posiciones, masas y códigos a la CPU, arma el octree con `construir_arbol()` de `c_core/` (`:306`) y lo sube aplanado en float. Las copias se miden dentro de esta fase, como pide el diseño.
3. **recorrido** (`kernel_recorrido`, `:135`): un hilo por partícula, pila de `PILA_MAX` enteros en memoria local (`:140`), sin recursión. Es el mismo algoritmo que `aceleracion_de()` de C1.
4. **integracion**: `kernel_medio_kick` y `kernel_drift`.

Cada fase se mide con un par de `cudaEvent` (`empezar` / `terminar`, `:66`): el tiempo es el de la GPU, incluidas las esperas.

**Decisiones y por qué:**
- **float32 en la GPU, double en la CPU**: la T4 hace FP32 mucho más rápido que FP64. El verificador usa umbrales de float32 (`--float32`). Los códigos Morton se calculan en double (`a_entero_gpu`) para que la celda de cada partícula coincida con la del árbol, que se arma en double.
- **`ORDEN_HILOS`** (`:39`, y la variable de entorno `ORDEN_HILOS=morton|original`): con `morton`, el hilo t atiende a la t-ésima partícula en orden Morton, así que los 32 hilos de un warp son vecinos en el espacio, abren casi las mismas celdas y el warp casi no diverge. Con `original`, el hilo t atiende a la partícula de id t (`hilo_a_particula`, `:108`), que está en cualquier parte: los hilos del warp recorren el árbol distinto y el warp tarda lo del más lento. La diferencia se ve en la fase `recorrido` del notebook.
- **Árbol en la CPU**: el enunciado pide árbol aplanado, partículas ordenadas y recorrido sin recursión, pero no dice dónde se construye. Está preguntado al profesor; el plan B es construirlo en la GPU (ver DISEÑO.md).
- **Sin memoria compartida ni `__syncthreads()`**: cada hilo tiene su pila local y escribe solo su partícula. Por eso `racecheck` no tiene nada que revisar y la prueba que importa es `memcheck`.
- **`REVISAR(...)`** envuelve cada llamada de CUDA y `cudaGetLastError()` va después de cada kernel: un error se reporta con archivo y línea en vez de dar resultados basura.

**Qué tendría que poder explicar el humano en la sustentación:**
- `blockIdx.x * blockDim.x + threadIdx.x` y por qué cada kernel empieza con `if (i >= n) return` (el último bloque tiene hilos de sobra).
- Divergencia de warp: qué es, por qué `morton` la reduce y dónde se cambia (`:39` o `ORDEN_HILOS`).
- Por qué la pila es local y no `__shared__`, y por qué entonces no hay carrera.
- Qué fases son secuenciales: la construcción en la CPU y las copias. Si dominan el desglose, el speedup de la GPU se aplana (Amdahl).
- Tamaño de bloque: `--workers` = hilos por bloque (64, 128, 256 en el notebook).
- Modificaciones de la demo: `ORDEN_HILOS`, `--workers`, `--theta`.

**Dudas abiertas:**
- ¿Acepta el profesor el árbol en la CPU?
- Resultados del notebook pendientes (verificador, memcheck y mediciones).
