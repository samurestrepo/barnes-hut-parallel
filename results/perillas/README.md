# Perillas de la demo probadas (2026-11-24)

`uniform` N = 20 000, 20 pasos, 4 trabajadores, salvo que se diga otra cosa. Error = mediana del error relativo de acc0 contra la suma directa, calculada aparte con `common/fisica.py`.

| Perilla | Comando | Resultado | Fuente |
| --- | --- | --- | --- |
| θ = 1.0 (mp) | `python py_mp/bh_mp.py --tree shared --workers 4 --theta 1.0 --input data/uniform_N20000_s42.txt --steps 20 --out X` | total 0.69 s frente a 1.81 s con θ = 0.5; 36.8 frente a 197.5 millones de interacciones (5.4× menos); error 1.65e-2 frente a 1.94e-3 | `mp_theta1_w4_*`, `mp_theta05_w4_*` |
| θ = 1.0 (MPI) | `mpiexec -n 4 python mpi/bh_mpi.py --theta 1.0 ...` | total 0.78 s frente a 1.91 s; bytes recibidos 11.4 MB (21.5% del árbol completo) frente a 16.3 MB (30.9%); error 1.83e-2 frente a 2.54e-3 | `mpi_theta1_n4_*`, `mpi_theta05_n4_*` |
| θ = 1.0 (S2) | `python py_core/bh_core.py --theta 1.0 ...` | total 0.96 s; error 1.65e-2 (igual que mp) | `s2_theta1_times.csv` |
| `--tree local` frente a `shared` | `python py_mp/bh_mp.py --tree local --workers 4 ...` | 1.87 s frente a 1.81 s con 4 procesos; 2.33 s frente a 1.92 s con 8 | `results/mp/uniform_N20000_*_bench.csv` |
| `--workers 8` (> 4 hilos) | `python py_mp/bh_mp.py --tree shared --workers 8 ...` | 1.92 s frente a 1.81 s con 4: sobresuscripción, no mejora; crear el Pool sube a ≈ 7 s | `results/mp/uniform_N20000_shared_bench.csv` |
| `-n 8` (MPI) | `mpiexec -n 8 python mpi/bh_mpi.py ...` | 2.27 s frente a 2.09 s con 4 (S4) o 2.57 s frente a 2.00 s (S5); sin `MPICH_PROGRESS_SPIN_LIMIT`, 9.6 s | `results/mpi/resumen_uniform_N20000.csv`, `results/mpi/sin_limite_giro/` |
| `--particion espacio` con `corner` | `mpiexec -n 4 python mpi/bh_mpi.py --particion espacio --input data/corner_N20000_s42.txt --dt 2.5e-4 --steps 20 ...` | 4.75 s frente a 2.40 s con `cantidad`; el rango 0 tiene 19 254 partículas | `results/mpi/corner_N20000_*_n4_*` |
| `TROZOS_POR_PROCESO = 1` con `corner` | cambiar `py_mp/bh_mp.py:40` | 20 pasos: igual de parejo (1.82–1.84 s por trabajador) | `results/mp/corner_N20000_shared_w4_trozos1_*` |
