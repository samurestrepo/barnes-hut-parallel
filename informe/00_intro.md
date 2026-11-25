---
title: "Barnes-Hut paralelo: multiprocessing, MPI, OpenMP y CUDA"
subtitle: "Programación Concurrente y Distribuida · Universidad de Caldas · 2026-2"
author:
  - Samuel Restrepo Ramírez
  - Sebastián Molina Escobar
date: "Noviembre de 2026"
lang: es
---

# Organización del proyecto

Las cuatro versiones comparten un **contrato** (`CONTRATO.md`): el mismo formato de entrada
(N × 7, `x y z vx vy vz m`), la misma física (G = 1, ε = 0.01, leapfrog kick-drift-kick), la misma
línea de comandos y las mismas salidas (aceleraciones del paso 0 y estado final en el orden de la
entrada, y `times.csv` con las fases lectura, construcción, ordenamiento, recorrido, integración y
comunicación). Con eso un solo verificador (`common/verify.py`) y un solo script de mediciones
(`common/bench.py`, `common/plot.py`) sirven para todas.

Las cuatro versiones usan el mismo esqueleto: caja raíz recalculada en cada paso, códigos de
Morton de 21 bits por eje, partículas ordenadas por código (cada celda del octree queda como un
tramo contiguo), octree en arreglos planos con hojas de hasta 8 partículas y recorrido con pila
explícita que abre siempre la celda que contiene a la partícula. Hay dos núcleos secuenciales, uno
por lenguaje: `py_core/` (Python con Numba, sin paralelismo propio) para multiprocessing y MPI, y
`c_core/` (C) para OpenMP y CUDA.

**Decisiones frente al enunciado.**

- **Hojas de hasta 8 partículas** en lugar de una: con 21 bits por eje dos partículas muy cercanas
  pueden compartir código y no se podrían separar; además, menos nodos significa menos memoria y
  un recorrido más corto. Dentro de una hoja la fuerza se suma partícula por partícula.
- **20 pasos para medir**, reportando tiempo por paso, en lugar de 100; los ensayos de balanceo
  sí usan 100 pasos. *(Pendiente de confirmar con el profesor.)*
- **dt = 2.5e-4 con distribuciones concentradas** (`corner`, `clusters`): con dt = 0.001 la energía
  no se conserva en `corner` aunque el código esté bien.
- **Árbol de CUDA construido en la CPU** con `c_core/`: el enunciado pide árbol aplanado, partículas
  ordenadas y recorrido sin recursión, pero no dice dónde se construye. *(Pendiente de confirmar
  con el profesor.)*

**Historial de git.** Una rama por implementación (`impl/py-core`, `impl/mp`, `impl/mpi`,
`impl/c-core`, `impl/omp`, `impl/cuda`) más ramas de herramientas y documentos. Cada una entró a
`main` por pull request **sin squash**, con la salida del verificador en la descripción
(PR #1 a #N). Un commit por tarea, con el código de la tarea al inicio del mensaje (`S2: …`,
`C3: …`). Los problemas del contrato se registraron en `CONTRATO_DUDAS.md` en lugar de cambiarlo
sin avisar. *(Actualizar #N y el conteo de commits al cerrar.)*
