# GUIA — Barnes-Hut con MPI (`mpi/bh_mpi.py`)

Una página para estudiar la versión sin reunirse. Detalle y números en `mpi/NOTAS.md`.

```
mpiexec -n 4 python mpi/bh_mpi.py --particion cantidad --input data/uniform_N20000_s42.txt --steps 20 --out salida/mpi
```
(En WSL2/OpenMPI: `mpirun --oversubscribe -n 4 python mpi/bh_mpi.py ...`)

## 1. Qué se paraleliza, por qué se puede y qué queda secuencial

- **Cada rango es dueño de un trozo contiguo en orden Morton** (≈ una región compacta del espacio). Calcula sus códigos, construye su árbol, poda, calcula las fuerzas de SUS partículas e integra solo esas. Se puede porque cada partícula solo necesita leer masas ajenas.
- **Ramas podadas**: en vez de replicar el árbol, cada rango manda a cada otro solo lo que necesita (`puntos_para_caja`, `mpi/bh_mpi.py:97`): las celdas que cumplen θ vistas desde el punto **más cercano** de la caja remota viajan como (cm, masa); las hojas cercanas viajan como partículas. Se recibe 27–33% de lo que pesaría el árbol completo.
- **Queda secuencial**: el rango 0 junta códigos y partículas, ordena (`:172`) y reparte, en cada paso, mientras los demás esperan; y la lectura y escritura de archivos.

## 2. Carreras y deadlocks: dónde podrían estar y cómo se evitaron

- **No hay memoria compartida**: cada rango tiene sus arreglos; no hay carreras de datos.
- **Deadlock por colectivas en distinto orden** (bug típico 2): las 11 colectivas de cada paso (numeradas en los comentarios `# SINCRONIZACIÓN:`) se llaman en el mismo orden en todos los rangos: Allreduce ×2 (`:134`), Gather + Gatherv ×2, Scatter + Scatterv ×2 (`:179`), Allgather (`:210`), Alltoall + Alltoallv (`:237`). El trabajo extra del rango 0 va entre dos colectivas, y un rango sin partículas igual las llama con búferes vacíos.
- **Deadlock por Send/Send** (bug típico 1): no hay envíos punto a punto; el intercambio es un `Alltoallv`.
- **Búfer de recepción mal dimensionado**: antes del `Alltoallv` va un `Alltoall` con las cuentas (`:233`), así cada uno reserva lo justo.
- **Caché de Numba escrita por varios procesos a la vez**: el 0 compila primero y los demás esperan en una barrera (`calentar`, `:330`).

## 3. Perillas de la demo

| Perilla | Dónde | Qué cambia |
| --- | --- | --- |
| particionado | `--particion cantidad\|espacio` (`mpi/bh_mpi.py:358`; defecto `:47`) | espacio con `corner`: un rango con 19 254 partículas y los otros 3 casi ociosos (4.75 s vs 2.40 s) |
| θ | `--theta` (`:355`); defecto `py_core/bh_core.py:29` | θ = 1.0: menos celdas abiertas y menos bytes enviados |
| rangos | `mpiexec -n K` | más de 4 = sobresuscripción en esta máquina |
| espera activa de MS-MPI | `MPICH_PROGRESS_SPIN_LIMIT` (`:29`) | con 0 (sin límite) y 8 rangos, 9.6 s en vez de 2.3 s |
| dt, ε | `--dt`, `--eps` (`:354`, `:356`) | con `corner`/`clusters` usar dt = 2.5e-4 |

## 4. Tres preguntas de predicción

1. **¿Qué pasa de 4 a 8 rangos?** Empeora un poco (2.09 → 2.27 s): 8 rangos en 4 hilos de hardware, y la comunicación sube de 0.30 a 0.94 s porque hay más rangos que esperar y más ramas que intercambiar (33 MB contra 16 MB).
2. **¿Más rangos = más bytes por rango?** No: la proporción frente al árbol completo baja (32.8% → 30.9% → 27.1% con 2, 4, 8), pero el total recibido sube porque hay más fronteras entre rangos.
3. **Con `corner` y `--particion espacio`, ¿qué rango tarda más y qué hacen los otros?** El 0 (tiene el rincón denso: 19 254 de 20 000 partículas, 4.2 s de recorrido); los otros terminan en 0.03 s y esperan dentro de la siguiente colectiva: su espera aparece como 4.16 s de "comunicación".

## 5. Qué se ve en el monitor de recursos

Medido con `wmic` cada ≈ 0.5 s, N = 20 000, 100 pasos, 4 rangos (`results/mpi/monitor_cpu.csv`; 100% = un hilo de hardware, máximo 400%):

- **Con `uniform` y `cantidad`**: 4 procesos al 60–95% (total 290–360%), con bajones cortos a ≈ 160–190% cuando todos esperan en las colectivas (el reparto en el rango 0).
- **Con `corner` y `cantidad`**: 4 procesos activos, pero uno siempre más bajo (35–55%) que los otros (80–95%): es el rango con las partículas sueltas, que tiene menos trabajo y espera. Es el desbalance de "cantidad no es trabajo" (15% menos interacciones en `workers.csv`).
- **Con `corner` y `espacio`**: durante ≈ 30 s **un solo proceso al ≈ 100%** y los otros tres en ≈ 0–10%; total ≈ 100% de 400%. Es el rango 0 con 19 254 partículas; los demás están dentro de una colectiva esperando. Con `MPICH_PROGRESS_SPIN_LIMIT=0` se ve casi igual (los que esperan al 10–20%), pero la comunicación sube: 10.9 s en 40 pasos (`results/mpi/monitor_cpu.csv`, escenario `..._sin_limite`).
- La corrida con `espacio` dura 35 s frente a 20 s con `cantidad`.
