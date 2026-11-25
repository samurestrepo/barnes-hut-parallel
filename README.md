# barnes-hut-parallel

Gravitational N-body simulation with the Barnes-Hut algorithm, parallelized four ways: Python multiprocessing, MPI (mpi4py), C with OpenMP, and CUDA. Includes a shared verifier, scaling benchmarks, and per-phase GPU timings.

Final project for *Programación Concurrente y Distribuida* (Concurrent and Distributed Programming), Universidad de Caldas, 2026-2.

## What it does

N point particles attract each other through gravity (G = 1, softened by ε). Each time step:

1. Build an **octree** over all particles; every cell stores its total mass and center of mass.
2. Compute the force on each particle by walking the tree: a cell that is far enough away (cell size / distance < θ) is treated as a single particle at its center of mass. This cuts the cost from O(N²) to O(N log N).
3. Update velocities and positions with the **leapfrog** (kick-drift-kick) integrator.

N ranges from 20,000 to 200,000 particles.

## Implementations

| Folder | Version | Parallelization challenge | Owner |
| --- | --- | --- | --- |
| `py_mp/` | Python + multiprocessing | Share or rebuild the tree across processes (shared memory vs. local rebuild, decided by measurement) | Samuel |
| `mpi/` | Python + mpi4py | Partition space with a Morton curve; each rank receives only the remote tree branches it needs | Samuel |
| `c_omp/` | C + OpenMP | Lock-free parallel tree construction from Morton order; load balancing across threads | Sebastián |
| `cuda/` | CUDA (Google Colab, T4) | Flattened tree, sorted particles, recursion-free traversal, low warp divergence | Sebastián |

Shared code lives in `common/` (input generator, verifier, benchmark scripts), `py_core/` (sequential Python tree) and `c_core/` (sequential flattened tree in C).

## Correctness

A version is correct when its accelerations match an O(N²) direct sum on small N within the error expected for its θ, its trajectory matches a direct integration, and total energy is conserved within tolerance. `common/verify.py` checks all of this for every version.

## Input format

Plain text, N rows × 7 columns: `x y z vx vy vz m`. Generate test inputs with:

```bash
python common/gen.py --dist uniform --n 20000 --seed 42
```

## How to run

Requirements: Python 3 with NumPy, Numba and mpi4py; OpenMPI (`mpirun`); gcc with OpenMP; for CUDA, Google Colab with a T4 GPU. On Windows, use WSL2 and keep the repository on the WSL disk (`~/`), not under `/mnt/c`: reading input from `/mnt/c` is about 35× slower and distorts every measurement.

All versions share the same command line: `--input F --steps S --dt 0.001 --theta 0.5 --eps 0.01 --workers K --out PREFIX`.

```bash
python common/gen.py --dist corner --n 20000                      # also: uniform, clusters --vel 0.3 --masas aleatorias
python py_core/bh_core.py --input data/corner_N20000_s42.txt --steps 20 --dt 2.5e-4 --out /tmp/s2
python py_mp/bh_mp.py --tree shared --workers 8 ...                # --tree shared | local
mpirun --oversubscribe -n 8 python mpi/bh_mpi.py --particion cantidad ...   # --particion cantidad | espacio
make -C c_omp && OMP_SCHEDULE="dynamic,64" ./c_omp/bh_omp --workers 8 ...   # OMP_SCHEDULE=static to compare
make -C cuda && ORDEN_HILOS=morton ./cuda/bh_cuda --workers 128 ...         # in Colab: cuda/bh_cuda.ipynb
```

Check a version with `python common/verify.py correr --cmd "<program>"` (add `--float32` for CUDA). Measure scaling with `common/bench.py` and plot it with `common/plot.py`.

Each implementation folder has a one-page `GUIA.md` (what is parallel, where races or deadlocks could occur, where each demo knob lives) and a `NOTAS.md` log. The report is assembled from `informe/` with `bash informe/armar.sh`.

## Status

All four versions pass the verifier and are merged into `main` (November 2026). Each one was developed on its own branch (`impl/<name>`) and merged through a pull request without squashing.

## Authors

- Samuel Restrepo Ramírez
- Sebastián Molina Escobar
