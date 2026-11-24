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

## Status

Work in progress (November 2026). Each implementation is developed on its own branch (`impl/<name>`) and merged into `main` once it passes the verifier.

## Authors

- Samuel Restrepo Ramírez
- Sebastián
