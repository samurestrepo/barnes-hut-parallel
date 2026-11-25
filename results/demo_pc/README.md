# Prueba de S2–S4 en el computador de la demo (2026-11-25)

Correr, verificar y medir las versiones de Samuel (S2 secuencial, S3 multiprocessing, S4 MPI) en el
portátil de la demo. No se cambió código: todo lo de esta carpeta son salidas, gráficas y este
README. El único script nuevo es `conjunta.py` (gráfica conjunta con un T1 por serie; usa las
funciones de `common/plot.py` sin modificarlo).

## Entorno

| | |
| --- | --- |
| Máquina | i7-11800H, 8 núcleos físicos / 16 hilos (`lscpu_demo.txt`, copia de `results/omp/lscpu_demo.txt`) |
| Sistema | WSL2, Ubuntu 26.04.1, kernel 6.18.58.1-microsoft-standard-WSL2, `nproc` = 16 |
| Python | 3.14.4 en un venv (`~/venv-bh`), numpy 2.5.5, numba 0.68.1, mpi4py 4.1.2, matplotlib 3.11.2 |
| MPI | Open MPI 5.0.10 (paquete de Debian/Ubuntu: `openmpi-bin`, `libopenmpi-dev`) |
| Repo | clon nuevo en `~/bh-demo` (disco de WSL, nunca `/mnt/c`), rama `tools/prueba-demo` desde `main` (4229930) |

Decisiones anotadas:
- `~/bh-medicion` (de C4) no era un clon de git sino una copia, así que se clonó de nuevo en `~/bh-demo`.
- `mpi4py` 4.1.2 se instala como rueda precompilada que busca `libmpi.so` al importarse. Se instaló
  antes que OpenMPI y luego se reinstaló (`pip install --force-reinstall --no-deps mpi4py`); funciona
  con OpenMPI 5.0.10 sin compilar nada.
- Los datos (`data/*.txt`), los `_acc0.txt` y los `_final.txt` no se suben (`.gitignore`).
  `verificador.txt` y `lscpu_demo.txt` se agregaron con `git add -f` porque `.gitignore` ignora `results/**/*.txt`.
- `results/lscpu.txt` (de Samuel) se restauró con `git checkout` después de cada `bench.py`.

## 1. Verificación (corrección)

Salida completa de cada corrida en `verificador.txt`. Valores (error medido < umbral):

| Versión | fuerzas_directa | fuerzas_bh | trayectoria | energia_reposo | energia_movimiento | Resultado |
| --- | --- | --- | --- | --- | --- | --- |
| S2 `bh_core.py` | 1.49e-14 | 3.25e-3 | 2.78e-17 | 3.37e-8 | 8.87e-3 | PASA |
| S3 `--tree shared` w1 / w4 / w8 | 1.49e-14 | 3.25e-3 | 2.78e-17 | 3.37e-8 | 8.87e-3 | PASA (las 3) |
| S3 `--tree local` w1 / w4 / w8 | 1.49e-14 | 3.25e-3 | 2.78e-17 | 3.37e-8 | 8.87e-3 | PASA (las 3) |
| S4 `mpirun -n 1` (1.ª corrida) | **FALLA** (código 1, sin mensaje) | 3.25e-3 | 2.78e-17 | 3.37e-8 | 8.87e-3 | FALLA |
| S4 `mpirun -n 1` (repetición) | 1.49e-14 | 3.25e-3 | 2.78e-17 | 3.37e-8 | 8.87e-3 | PASA |
| S4 `mpirun -n 2` | 1.49e-14 | 3.25e-3 | 2.78e-17 | 3.37e-8 | 8.90e-3 | PASA |
| S4 `mpirun -n 4` | 1.49e-14 | 3.24e-3 | 2.78e-17 | 3.37e-8 | 8.98e-3 | PASA |
| S4 `mpirun -n 8` | 1.49e-14 | 3.42e-3 | 2.78e-17 | 3.37e-8 | 8.86e-3 | PASA |

Comparación con Windows: S2 y S3 dan exactamente lo mismo (3.25e-3 y 8.87e-3). S4 con 1, 2 y 4
rangos da los mismos valores que `mpi/NOTAS.md` (3.25e-3 / 3.25e-3 / 3.24e-3 y 8.87e-3 / 8.90e-3 /
8.98e-3); con θ = 0 es idéntico a S2. Con 8 rangos (no medido en Windows) `fuerzas_bh` sube a
3.42e-3, lejos del umbral 1e-2.

**La falla de S4 con 1 rango (no se arregló nada):**
- Comando: `python common/verify.py correr --cmd "mpirun --oversubscribe -n 1 python mpi/bh_mpi.py"`
  (2026-11-25 15:27, primera ejecución de `mpi/bh_mpi.py` en este computador).
- Error completo, tal como lo imprime el verificador: `fuerzas_directa FALLA nan < 1e-06 salió con código 1: `
  (ni stderr ni stdout tenían texto). Las otras 4 pruebas de esa misma corrida pasaron.
- No se reprodujo: 3 corridas sueltas del mismo comando (`--steps 1 --theta 0.0`, N = 2000), una más
  borrando antes la caché de Numba de `mpi/__pycache__`, y la repetición completa del verificador
  (al final de `verificador.txt`) terminaron bien.
- Hipótesis (sin comprobar): fue el primer `mpirun` después de instalar OpenMPI; puede haber fallado
  el arranque de OpenMPI (PRRTE crea su directorio de sesión en `/tmp` la primera vez) antes de
  llegar a Python, por eso no hay traceback. No apunta a un error de `bh_mpi.py`.
- **Para la demo**: correr una vez cualquier `mpirun` de prueba antes de empezar.

## 2. Escalado

`uniform`, θ = 0.5, dt = 1e-3, 20 pasos; N = 20 000 con 3 repeticiones (promedio) y N = 200 000 con 1.
T1 = S2 con 3 repeticiones (`secuencial/`): **1.922 s** con N = 20 000 (1.948, 1.942, 1.877) y
**28.343 s** con N = 200 000 (29.06, 28.24, 27.73). Gráficas en `<serie>_N<N>_escalado.png` y los
números en `<serie>_N<N>_escalado.csv`; corridas sueltas en `<serie>/bench_uniform_N*/`.
Ninguna corrida pasó de 20 min (la más larga: 30 s), así que se midió también 32.

**mp shared** (`python py_mp/bh_mp.py --tree shared`):

| Procesos | N=20k: s | Speedup | Eficiencia | Karp-Flatt | N=200k: s | Speedup | Eficiencia | Karp-Flatt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.86 | 1.04 | 1.04 | — | 27.51 | 1.03 | 1.03 | — |
| 2 | 1.04 | 1.85 | 0.92 | 0.08 | 15.28 | 1.86 | 0.93 | 0.08 |
| 4 | 0.64 | 3.01 | 0.75 | 0.11 | 9.06 | 3.13 | 0.78 | 0.09 |
| 8 | 0.47 | 4.12 | 0.52 | 0.13 | 6.67 | 4.25 | 0.53 | 0.13 |
| 16 | 0.54 | 3.56 | 0.22 | 0.23 | 5.10 | 5.55 | 0.35 | 0.13 |
| 32 | 0.74 | 2.59 | 0.08 | 0.37 | 5.23 | 5.42 | 0.17 | 0.16 |

**mp local** (`python py_mp/bh_mp.py --tree local`):

| Procesos | N=20k: s | Speedup | Eficiencia | Karp-Flatt | N=200k: s | Speedup | Eficiencia | Karp-Flatt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.87 | 1.03 | 1.03 | — | 27.31 | 1.04 | 1.04 | — |
| 2 | 1.04 | 1.85 | 0.92 | 0.08 | 15.48 | 1.83 | 0.92 | 0.09 |
| 4 | 0.68 | 2.83 | 0.71 | 0.14 | 9.48 | 2.99 | 0.75 | 0.11 |
| 8 | 0.55 | 3.52 | 0.44 | 0.18 | 7.10 | 3.99 | 0.50 | 0.14 |
| 16 | 0.61 | 3.13 | 0.20 | 0.27 | 7.34 | 3.86 | 0.24 | 0.21 |
| 32 | 1.02 | 1.89 | 0.06 | 0.51 | 9.55 | 2.97 | 0.09 | 0.32 |

**MPI** (`mpirun --oversubscribe -n {k} python mpi/bh_mpi.py`, `--particion cantidad` por defecto):

| Rangos | N=20k: s | Speedup | Eficiencia | Karp-Flatt | N=200k: s | Speedup | Eficiencia | Karp-Flatt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.85 | 1.04 | 1.04 | — | 28.08 | 1.01 | 1.01 | — |
| 2 | 1.14 | 1.68 | 0.84 | 0.19 | 16.50 | 1.72 | 0.86 | 0.16 |
| 4 | 0.74 | 2.58 | 0.65 | 0.18 | 10.68 | 2.65 | 0.66 | 0.17 |
| 8 | 0.61 | 3.13 | 0.39 | 0.22 | 7.93 | 3.58 | 0.45 | 0.18 |
| 16 | 0.70 | 2.75 | 0.17 | 0.32 | 6.28 | 4.51 | 0.28 | 0.17 |
| 32 | 1.25 | 1.54 | 0.05 | 0.64 | 8.70 | 3.26 | 0.10 | 0.28 |

Bytes recibidos frente al árbol completo (N = 200 000, `bytes_*` de `times.csv`): 15.4% (2),
17.7% (4), 15.6% (8), 11.4% (16), 8.3% (32); los de 2, 4 y 8 son iguales a los de Windows.

**Gráfica conjunta** (`conjunta_N20000_escalado.png` y `conjunta_N200000_escalado.png`): mp shared,
mp local, MPI y OpenMP (`results/omp/uniform_N*_bench.csv`), cada una contra su secuencial: S2 para
Python (1.922 / 28.343 s) y C1 para OpenMP (3.69 / 45.46 s, los de C4).

Lectura (medido):
- Hasta 8 (núcleos físicos) las tres escalan; mp shared es la mejor de Python y en eficiencia
  queda a la par de OpenMP (N = 200 000, 8: 0.53 contra 0.56).
- Con N = 20 000 todas las de Python empeoran de 8 a 16 y más aún a 32. Crear el `Pool` cuesta
  0.43 s con 1 proceso, 0.51 s con 8, 0.70 s con 16 y 1.2 s con 32 (fila `crear_pool`); no entra en
  `total`, pero el arranque de los procesos y del intercambio sí pesa frente a ≈ 0.5 s de cálculo.
- Con N = 200 000, mp shared sigue mejorando hasta 16 (5.10 s, hyperthreading) y se queda igual con
  32; mp local empeora desde 8 porque cada proceso reconstruye el árbol: la fase `construccion`
  pasa de 0.75 s (1) a 1.36 s (8), 2.68 s (16) y 4.40 s (32).
- MPI tiene Karp-Flatt ≈ 0.17 casi constante de 2 a 16 con N = 200 000 (fracción secuencial: el
  rango 0 junta, ordena y reparte) y sube a 0.28 con 32. Su comunicación con N = 20 000 (rep. 2):
  0.05 s (2), 0.13 (4), 0.18 (8), 0.28 (16), 0.71 (32).
- **`MPICH_PROGRESS_SPIN_LIMIT` con OpenMPI**: no se vio la comunicación anormal de Windows (allá,
  8 rangos en 4 hilos daban ≈ 8 s de comunicación). Aquí, con 32 rangos en 16 hilos, son 0.71 s
  (N = 20 000) y 3.79 s de 8.70 (N = 200 000): crece con la sobresuscripción, pero sin el salto de
  Windows.

## 3. Ensayo de la demo

Igual que `results/demo/README.md` pero con 8 trabajadores: N = 20 000, dt = 2.5e-4, θ = 0.5,
ε = 0.01, 100 pasos. Archivos en `ensayo/` (`_times.csv` y `_workers.csv`).

| Datos | Versión | Total (s) | Recorrido (s) | Comunicación (s) | Deriva de energía | Trabajadores: segundos (mín – máx) |
| --- | --- | --- | --- | --- | --- | --- |
| clusters | S2 secuencial | 15.73 | 15.33 | 0 | 1.273e-2 | — |
| clusters | mp `--tree shared` w8 | 3.72 | 3.15 | 0.02 | 1.273e-2 | 2.76 – 2.79 |
| clusters | MPI `cantidad` n8 | 5.37 | 3.89 | 2.03 | 1.286e-2 | 2.54 – 3.89 |
| corner | S2 secuencial | 13.20 | 12.80 | 0 | 4.567e-4 | — |
| corner | mp `--tree shared` w8 | 3.08 | 2.53 | 0.02 | 4.567e-4 | 2.21 – 2.22 |
| corner | MPI `cantidad` n8 | 4.85 | 3.13 | 2.60 | 9.941e-4 | 1.45 – 3.13 |
| corner | MPI `espacio` n8 | 20.29 | 19.14 | 19.73 | 4.005e-4 | 0.04 – 19.14 |

Speedup contra S2: mp 4.23 (clusters) y 4.29 (corner); MPI `cantidad` 2.93 y 2.72; MPI `espacio` 0.65.

- mp con trozos dinámicos deja a los 8 trabajadores dentro del 1%; MPI con `cantidad` va de 1.45 a
  3.13 s con `corner` (94 frente a 223 millones de interacciones): los demás esperan al más lento y
  eso aparece como comunicación (2.60 s). Es el mismo efecto que en Windows con 4 rangos, más marcado.
- Con `espacio` el rango 0 se queda con ≈ 19 120 de las 20 000 partículas por paso y hace 1 419
  millones de interacciones; los otros 7 terminan en 0.04 s y esperan 19 s. Es la demostración más
  visible de por qué `espacio` no sirve con `corner`.
- La deriva de S2 y mp es idéntica a Windows (1.273e-2 y 4.567e-4). La de MPI cambia con el número
  de rangos (n4 en Windows: 1.278e-2 y 7.612e-4; n8 aquí: 1.286e-2 y 9.941e-4) porque el árbol
  combinado aproxima distinto; todas quedan bajo el umbral del verificador (2e-2).

**Perillas** (corner, N = 20 000, 20 pasos, dt = 2.5e-4; `perillas/`). Todas funcionan:

| Corrida | Total (s) | Detalle |
| --- | --- | --- |
| mp shared w8, θ = 0.5 | 0.548 | trabajadores 0.36 s, ≈ 30 millones de interacciones cada uno |
| mp shared w8, `--theta 1.0` | 0.539 | trabajadores 0.09 s, ≈ 6.7 millones cada uno |
| mp `--tree local` w8 | 0.635 | construcción 0.116 s (shared: 0.021) |
| mp shared `--workers 32` | 0.831 | `crear_pool` 1.12 s |
| MPI n8, θ = 0.5 | 0.828 | comunicación 0.26 s |
| MPI n8, `--theta 1.0` | 0.436 | comunicación 0.16 s |
| MPI n8, `--particion espacio` | 3.503 | comunicación 3.05 s; rango 0 con ≈ 19 120 partículas |
| MPI `-n 32` | 1.807 | comunicación 1.33 s |

## Comparación con Windows (i5-6300U, 2 núcleos / 4 hilos)

| Medida | Windows (`results/mp`, `mpi`, `demo`) | PC de la demo |
| --- | --- | --- |
| T1 de S2 implícito, N = 20 000 (speedup 1 → tiempo) | ≈ 3.3 s | 1.92 s |
| Mejor mp shared, N = 20 000 | 1.81 s con 4 (S = 1.84) | 0.47 s con 8 (S = 4.12) |
| Mejor mp shared, N = 200 000 | 24.8 s con 4–8 (S = 2.11) | 5.10 s con 16 (S = 5.55) |
| Mejor MPI, N = 20 000 | 2.00 s con 4 (S = 1.66) | 0.61 s con 8 (S = 3.13) |
| Mejor MPI, N = 200 000 | 26.8 s con 4 (S = 1.95) | 6.28 s con 16 (S = 4.51) |
| Ensayo clusters: S2 / mp / MPI | 29.02 / 15.91 (w4) / 16.62 (n4) s | 15.73 / 3.72 (w8) / 5.37 (n8) s |
| Ensayo corner: S2 / mp / MPI | 25.15 / 12.76 (w4) / 14.08 (n4) s | 13.20 / 3.08 (w8) / 4.85 (n8) s |
| Crear el `Pool` | 1.8 a 7.2 s (*spawn*) | 0.43 a 1.2 s (*fork* en Linux) |

El patrón es el mismo (shared ≥ local, mp > MPI, MPI `cantidad` desbalanceado con `corner`); aquí
hay 4 veces más núcleos y se nota en todos los speedups.

## Comandos para la demo

Desde Ubuntu (WSL), siempre en el disco de WSL:

```bash
source ~/venv-bh/bin/activate && cd ~/bh-demo && mkdir -p /tmp/demo
mpirun --oversubscribe -n 2 python -c "from mpi4py import MPI"   # calentar OpenMPI (ver la falla de arriba)
D=data/corner_N20000_s42.txt   # o data/clusters_N20000_s42_v_m.txt
python py_core/bh_core.py --input $D --steps 100 --dt 2.5e-4 --theta 0.5 --out /tmp/demo/s2
python py_mp/bh_mp.py --tree shared --workers 8 --input $D --steps 100 --dt 2.5e-4 --theta 0.5 --out /tmp/demo/mp
mpirun --oversubscribe -n 8 python mpi/bh_mpi.py --particion cantidad --input $D --steps 100 --dt 2.5e-4 --theta 0.5 --out /tmp/demo/mpi
mpirun --oversubscribe -n 8 python mpi/bh_mpi.py --particion espacio --input $D --steps 100 --dt 2.5e-4 --theta 0.5 --out /tmp/demo/mpi_esp
python common/verify.py deriva --input $D --final /tmp/demo/mp_final.txt
cat /tmp/demo/mp_workers.csv /tmp/demo/mpi_workers.csv
```

Para una corrida rápida de perillas, `--steps 20` y cambiar `--theta 1.0`, `--tree local`,
`--workers 32` o `-n 32`. Si se corre como root, agregar `--allow-run-as-root` a `mpirun`.

## Cosas raras

Medido:
1. La primera verificación de S4 con 1 rango falló en `fuerzas_directa` con código 1 y sin mensaje;
   no se volvió a reproducir (ver la sección 1).
2. **El secuencial de Python es más rápido que el de C** en esta máquina: 1.92 s frente a 3.69 s
   (N = 20 000) y 28.3 s frente a 45.5 s (N = 200 000). Por eso en la gráfica conjunta OpenMP queda
   por encima en tiempo aunque escale igual o mejor.
3. En mp, `--theta 1.0` baja el tiempo de los trabajadores de 0.36 a 0.09 s, pero la fase
   `recorrido` de `times.csv` casi no cambia (0.419 → 0.423 s) y el total tampoco (0.548 → 0.539 s).
4. `mpirun --version` imprime "Sorry! You were supposed to get help about: version / But I couldn't
   open the help file" (faltan archivos de ayuda de PMIx/PRRTE en el paquete de Ubuntu). `mpirun`
   funciona normalmente.
5. Con 1 trabajador, mp y MPI dan un poco menos que T1 (speedup 1.01–1.04). Igual que en C4.

Hipótesis (sin comprobar):
- (2) Numba compila con LLVM y optimizaciones agresivas, y `c_core` se compiló con `gcc -O2`;
  además el ordenamiento es distinto (`np.argsort` frente a `qsort` con comparador). Habría que
  perfilar la fase `recorrido` de los dos para saberlo.
- (3) La fase `recorrido` de mp mide el tiempo de pared alrededor del reparto de trozos, que incluye
  ≈ 0.3 s fijos de despacho entre procesos (o la carga de la caché de Numba en los trabajadores en el
  primer paso). Con solo 20 pasos ese costo fijo tapa la mejora de θ; con 100 pasos debería notarse.
- (5) Turbo del portátil y caché caliente después de las corridas de T1.
