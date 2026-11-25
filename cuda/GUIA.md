# GUIA — Barnes-Hut en CUDA (`cuda/bh_cuda.cu`)

Una página para estudiar la versión sin reunirse. Detalle en `cuda/NOTAS.md`; se corre con `cuda/bh_cuda.ipynb` en Colab (GPU T4).

```
make -C cuda
./cuda/bh_cuda --input data/uniform_N200000_s42.txt --steps 20 --workers 128 --out salida/cuda
ORDEN_HILOS=original ./cuda/bh_cuda ...        # asignación a warps sin recompilar
python common/verify.py correr --cmd "./cuda/bh_cuda" --float32 --workers 128
```

## 1. Qué se paraleliza, por qué se puede y qué queda secuencial

- **Recorrido** (`kernel_recorrido`, `cuda/bh_cuda.cu:135`): un hilo por partícula. Cada hilo lee el árbol y escribe solo su aceleración. La pila (`:140`) es un arreglo local de cada hilo: no hay recursión.
- **Códigos Morton** (`kernel_codigos`, `:94`), **ordenamiento** (`thrust::sort_by_key`, `:248`) y **mover las partículas** (`thrust::gather` en `reordenar`, `:235`): en la GPU.
- **Integración** (`kernel_medio_kick`, `kernel_drift`, `:160`): un hilo por partícula.
- **Secuencial, en la CPU**: la construcción del octree (`construir_en_cpu`, `:299`) con `c_core/`, más las copias GPU→CPU→GPU. Todo eso se mide como fase `construccion`. Si esa fase pesa mucho frente a `recorrido`, el speedup se aplana (Amdahl).
- Cada fase se mide con `cudaEvent` (`:66`).

## 2. Carreras, deadlocks y errores de memoria

- **Sin memoria compartida (`__shared__`) ni `__syncthreads()`**: cada hilo usa su propia pila y escribe su propia partícula. No hay carreras ni puede haber deadlock de barreras.
- **Hilos de sobra** en el último bloque: cada kernel empieza con `if (i >= n) return`.
- **`hilo_a_particula[id[k]] = k`** (`:108`): sin carrera porque los `id` son una permutación (cada hilo escribe una casilla distinta).
- **Errores de CUDA**: `REVISAR(...)` en cada llamada y `cudaGetLastError()` después de cada kernel (`:246`, `:327`).
- **compute-sanitizer**: `memcheck` (accesos fuera de los arreglos) es la prueba que importa; `racecheck` solo mira `__shared__`, que no se usa.
- **Bugs típicos que NO están aquí pero pueden salir en papel**: reducción en `__shared__` sin `__syncthreads()`; `__syncthreads()` dentro de un `if` divergente (deadlock: unos hilos esperan en una barrera a la que otros nunca llegan); `+=` sobre memoria global desde varios hilos sin `atomicAdd`.

## 3. Perillas de la demo

| Perilla | Dónde | Qué cambia |
| --- | --- | --- |
| asignación a warps | `ORDEN_HILOS=morton\|original` o `ORDEN_HILOS_POR_DEFECTO` (`:39`) | `original`: un warp mezcla partículas lejanas y diverge; sube `recorrido` |
| hilos por bloque | `--workers 64\|128\|256` | ocupación de la GPU |
| θ | `--theta` | θ = 1.0: menos celdas abiertas |
| dt, ε | `--dt`, `--eps` | con `corner`/`clusters` usar dt = 2.5e-4 |

## 4. Tres preguntas de predicción

1. **¿Qué pasa con `ORDEN_HILOS=original`?** Sube el tiempo de `recorrido`: los 32 hilos de un warp ejecutan la misma instrucción a la vez; si cada uno recorre el árbol distinto, el warp hace la unión de todos los caminos y espera al más largo. Con `morton` los vecinos abren casi las mismas celdas. Medido con N = 200 000: 1.77 → 6.28 s (`uniform`), 3.6 veces más lento.
2. **¿Qué fase domina con N = 200 000?** El `recorrido` (1.77 s de 2.30 s con `morton`). La `construccion` en la CPU con las copias es la segunda (0.27 s, ≈ 12%): es la parte secuencial, y por Amdahl limita la mejora a ≈ 8 aunque el recorrido fuera instantáneo.
3. **¿64, 128 o 256 hilos por bloque?** Casi igual: recorrido 1.88, 1.81 y 1.80 s. El recorrido está limitado por las lecturas dispersas del árbol y la pila en memoria local, no por la ocupación.

## 5. Qué se ve en el monitor

`nvidia-smi` en otra celda (o `!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1`): la GPU sube a casi 100% durante el recorrido y baja a ≈ 0% mientras la CPU construye el árbol; ese ir y venir es la fracción secuencial. Sin Colab en la demo, se explica con el desglose de `results/cuda/*_times.csv`.
