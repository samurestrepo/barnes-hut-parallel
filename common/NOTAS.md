# Notas de `common/` (S0, S1, S5)

Bitácora de lo que se construyó, para estudiarlo en otro chat. Las entradas nuevas van al final.

---

## S0 — estructura y generador (2026-11-24)

**Archivos creados:** `CLAUDE.md`, `CONTRATO.md` (copia literal de la sección "Contrato común" de `DISEÑO.md`), `CONTRATO_DUDAS.md`, `DISEÑO.md`, `common/gen.py`, `.gitkeep` en `data/ py_core/ py_mp/ mpi/ c_core/ c_omp/ cuda/ results/ informe/`. El único dato que se sube es `data/uniform_N2000_s42.txt`; los demás se regeneran.

**Decisiones y por qué:**
- `gen.py` es el código del contrato con un solo cambio: escribe siempre en `<repo>/data` (`common/gen.py:13`), no en `data/` relativo a donde se llame. Así cualquiera lo corre desde cualquier carpeta y el nombre del archivo es el del contrato.
- La semilla fija (`common/gen.py:25`) hace que los datos sean reproducibles: dos generaciones dan el mismo archivo (mismo md5). Por eso los datos grandes no se suben.
- Se comprobó que las tres distribuciones con N = 2000 y 20 000 tienen forma N×7 y masa total 1.

**Qué tendría que poder explicar el humano:**
- Las tres distribuciones: `uniform` (la pide el enunciado), `clusters` (`common/gen.py:28`, dos cúmulos gaussianos con σ = 0.03 que, con `--vel`, se mueven uno hacia el otro) y `corner` (`common/gen.py:31`, el 95% de las partículas en el cubo [0, 0.05]³). `corner` castiga el reparto por espacio: si se parte el cubo en P pedazos iguales, un proceso se queda con casi todas las partículas.
- La masa total se normaliza a 1 (`common/gen.py:40`) para que la energía tenga la misma escala con cualquier N y las derivas se puedan comparar.
- Los sufijos `_v`, `_m` y `_e` del nombre indican velocidades, masas aleatorias y escala distinta de 1.
- Aquí no hay concurrencia, así que no hay carreras ni deadlocks.

**Dudas abiertas:** ninguna.

---

## S1 — verificador (2026-11-24)

**Archivos creados:**
- `common/fisica.py`: suma directa O(N²) por bloques, energía total y leapfrog de referencia.
- `common/verify.py`: tres modos, `correr`, `revisar` y `deriva`.
- `common/directo.py`: versión de referencia con el CLI del contrato. Sirve para probar el verificador.

**Archivos cambiados:** `CONTRATO_DUDAS.md` (dos dudas). El 24-nov se cambiaron también `CONTRATO.md` (formato `%.17g`), `CONTRATO_DUDAS.md` (la duda del formato, resuelta) y `CLAUDE.md` (rol del agente y cierre de cada tarea).

**Resultado al cerrar:**

| Prueba | Referencia (`directo.py`) | Sin segundo medio kick |
| --- | --- | --- |
| fuerzas_directa | PASA, error 0 | PASA, error 0 |
| fuerzas_bh | PASA, error 0 | PASA, error 0 |
| trayectoria | PASA, error 0 | FALLA, 9.2e-5 |
| energia_reposo | PASA, 3.4e-8 | FALLA, 5.6e-3 |
| energia_movimiento | PASA, 4.0e-4 | FALLA, 2.0e-1 |

También se probó a mano: una salida desordenada FALLA en la trayectoria (1.55), un archivo que falta da FALLA con el mensaje "no existe…", y una salida redondeada a float32 PASA con `--float32` (5.8e-8).

**Decisiones y por qué:**
- **El verificador corre la versión él mismo** (`correr --cmd`, `common/verify.py:156`). Así las cuatro versiones se prueban con los mismos datos y parámetros, sin que nadie los copie a mano. MPI se prueba con `--cmd "mpirun -n 4 python mpi/bh_mpi.py"`. Para Colab existe `revisar`, que mide salidas ya escritas.
- **Cada prueba corre la versión una vez y mide una sola cosa** (tabla `PRUEBAS`, `common/verify.py:39`). Así cuesta un poco más, pero se lee fácil y cada FALLA apunta a una sola causa.
- **La suma directa va por bloques de filas** (`common/fisica.py:12` y `:23`): con N = 20 000, la matriz completa de diferencias ocuparía unos 10 GB.
- **La física de referencia está en `fisica.py`**, separada del verificador, para que `verify.py` y `directo.py` usen exactamente el mismo código.
- **`times.csv` solo genera AVISO, no FALLA** (`common/verify.py:83`): mide rendimiento, no corrección.
- **Las salidas se escriben con `%.17g`** (`common/directo.py:19`): con `%.6e`, el redondeo del texto rozaba el umbral de 1e-6. Quedó fijado en `CONTRATO.md`.

**Qué tendría que poder explicar el humano:**
- **Las 5 pruebas y qué bug atrapa cada una:**
  - `fuerzas_directa`: con θ = 0 Barnes-Hut debe dar la suma directa salvo redondeo, y se mide el error relativo máximo (`common/verify.py:104`).
  - `fuerzas_bh`: con θ = 0.5 se mide la mediana (`:109`), porque las partículas con aceleración casi nula tienen un error relativo enorme aunque el código esté bien.
  - `trayectoria`: compara posición por posición contra 10 pasos integrados con suma directa (`:115`). Atrapa un integrador mal hecho y también una salida que quedó en orden Morton. La energía no atrapa lo segundo, porque no cambia si se reordenan las partículas.
  - Las dos de energía miden |E_f − E_0| / |E_0| (`:125`).
- **El leapfrog kick-drift-kick** (`common/fisica.py:69`): medio kick con la aceleración vieja, drift completo, recálculo de la aceleración y medio kick con la nueva. Sin el segundo medio kick (`common/directo.py:50`), las fuerzas del paso 0 siguen bien (por eso las dos primeras pruebas PASAN), pero la velocidad acumula solo la mitad de los kicks y la energía deriva.
- **La energía** (`common/fisica.py:55`): la suma recorre todos los pares (i, j) con j ≠ i, así que cada par cuenta dos veces. Por eso se divide entre 2 (`:66`). La diagonal se anula (`:37`) porque una partícula no se atrae a sí misma; con ε = 0 sería 0/0.
- **Por qué con la nube en reposo el umbral es tan estricto (1e-5):** 100 pasos × 0.001 = 0.1 unidades de tiempo, y el colapso tarda unas 0.5, así que las partículas casi no se mueven. Un integrador correcto da 3.4e-8; con un umbral de 1e-2 pasaba hasta Euler.
- Aquí no hay concurrencia: `correr` lanza la versión con `subprocess.run`, una prueba a la vez y esperando a que termine. No hay carreras ni deadlocks.

**Dudas abiertas:**
- El umbral de `energia_movimiento` (1e-3) es una suposición. La suma directa sola ya da 4.0e-4, y Barnes-Hut con θ = 0.5 suma su propio error. Se ajusta con S2 y se deja fijo (riesgo 3 del diseño).
- `fuerzas_bh` no tiene medida propia para float32: usa 1e-2 en los dos casos, porque el error de θ = 0.5 domina sobre el de float32.
- Una prueba completa tarda unos 60 s con `directo.py`, casi todo en las dos de energía de 100 pasos. Con Barnes-Hut debería ser menos.

---

## S5 — mediciones y gráficas (2026-11-24)

**Rama:** `tools/bench`, creada desde `impl/py-core` y con `impl/mp` e `impl/mpi` mergeadas (hacen falta para medir). Como ningún PR anterior está en `main` (umbral de `energia_movimiento`), el PR de S5 muestra también S2–S4 hasta que esos se mergeen.

**Archivos creados:**
- `common/bench.py`: bucle de mediciones. Corre una versión con 1, 2, 4, … hasta 2× los núcleos lógicos (`trabajadores_por_defecto`, `common/bench.py:32`), guarda la descripción del procesador en `results/lscpu.txt` (`guardar_lscpu`, `:42`) y escribe `results/<impl>/<etiqueta>_bench.csv` más las salidas de cada corrida en `results/<impl>/bench_<etiqueta>/`.
- `common/plot.py`: lee uno o más `*_bench.csv` y dibuja tiempo, speedup, eficiencia y Karp-Flatt en un PNG de 4 paneles, más un CSV con los números (`<out>_escalado.png/.csv`).
- `results/lscpu.txt` (queda solo en local: el `.gitignore` excluye `results/**/*.txt`; su contenido está abajo, en "`lscpu`"); `results/py_core/uniform_N200000_times.csv`; `results/mp/uniform_N{20000,200000}_{shared,local}_bench.csv`; `results/mpi/uniform_N{20000,200000}_cantidad_bench.csv`; las gráficas `results/mp/uniform_N*_escalado.png` (shared y local) y `results/mpi/uniform_N*_escalado.png` (las tres series), con sus CSV.

**Cómo lo usa Sebastián** (solo cambia el comando; `--workers` lo interpreta cada versión):
```
python common/bench.py --impl omp --etiqueta uniform --cmd "./c_omp/bh_omp"
python common/bench.py --impl cuda --etiqueta bloques --cmd "./cuda/bh_cuda" --trabajadores 64,128,256
python common/plot.py --out results/omp/uniform --base <T secuencial> --serie omp=results/omp/uniform_bench.csv
```
Con MPI en OpenMPI: `--lanzador "mpirun --oversubscribe -n {k}"`; en Windows (MS-MPI): `--lanzador "mpiexec -n {k}"`.

**Resultados** (`uniform`, θ = 0.5, 20 pasos; base = S2 secuencial; `results/mpi/uniform_N*_escalado.csv`):

| N | Versión | p = 1 | p = 2 | p = 4 | p = 8 |
| --- | --- | --- | --- | --- | --- |
| 20 000 (T1 = 3.315 s) | mp shared | S 0.90 | 1.31 (E 0.66) | 1.84 (E 0.46) | 1.73 (E 0.22) |
| | mp local | 0.95 | 1.31 (0.65) | 1.77 (0.44) | 1.43 (0.18) |
| | mpi cantidad | 0.88 | 1.36 (0.68) | 1.66 (0.42) | 1.29 (0.16) |
| 200 000 (T1 = 52.38 s) | mp shared | 0.97 | 1.55 (0.78) | 2.11 (0.53) | 2.11 (0.26) |
| | mp local | 0.98 | 1.62 (0.81) | 2.07 (0.52) | 1.77 (0.22) |
| | mpi cantidad | 0.97 | 1.43 (0.71) | 1.95 (0.49) | 1.94 (0.24) |

Ninguna corrida con N = 200 000 pasó de 30 min: la más larga fue 60 s de pared (S2 52.4 s de simulación). Bytes MPI con N = 200 000: 27.9 MB de 181.0 MB (15.4%) con 2 rangos, 95.8 de 543.0 (17.7%) con 4 y 197.7 de 1267.2 (15.6%) con 8.

**Lectura de los números:**
- **Con N grande escala mejor** (2.11 contra 1.84 con 4 procesos): el trabajo paralelo (recorrido) crece como N log N y la parte fija (crear procesos, colectivas, ordenar) pesa menos. Es el argumento de Gustafson.
- **Karp-Flatt ≈ 0.3–0.5 y sube de 4 a 8** en todas las versiones. Si la fracción secuencial real fuera ≈ 0.03–0.06 (lo medido en S2), e debería ser plano y pequeño. Lo que mide aquí es el **hardware**: 2 núcleos físicos con hyper-threading se comportan como algo menos de 3 núcleos, y eso Karp-Flatt lo ve como "fracción secuencial". La subida a 8 es el overhead de sobresuscribir (8 procesos en 4 hilos): más construcciones repetidas en local, más colectivas en MPI.
- **Recorrido con 4 y 8 procesos casi igual** con N = 200 000 (shared: 22.2 s y 21.8 s): con 4 hilos de hardware ya están todos ocupados.
- **En MPI la comunicación crece con p** (N = 200 000: 0.17, 1.66, 2.12, 3.54 s con 1, 2, 4, 8): mover 200 000 partículas al rango 0 y de vuelta en cada paso, más las ramas, más la espera por el rango más lento. El ordenamiento del rango 0 se queda en ≈ 0.33 s (1.2% de 27 s).
- **Variación entre tandas** (para no confundirla con un efecto): la misma configuración medida en dos momentos distintos dio, por ejemplo, mp local con 1 proceso 5.09 s (S3) y 3.49 s (S5), y mp shared con 4 procesos 2.04 s y 1.81 s. Es un portátil (turbo y temperatura). Las comparaciones entre versiones se hacen dentro de la misma tanda (la de S5).

**Decisiones y por qué:**
- **Salidas de cada corrida en una subcarpeta `bench_<etiqueta>/`** (`correr_una`, `:63`): la primera versión las escribía con el mismo nombre que las mediciones de S3 (`uniform_N20000_shared_w1_r1…`) y las pisó en la copia de trabajo; se restauraron desde git y se cambió el nombre.
- **`--lanzador` con `{k}`** (`:66`) en vez de detectar MPI: el mismo script sirve para OpenMPI, MS-MPI y las versiones que no usan lanzador.
- **Una sola figura de 4 paneles** con un eje cada uno (nunca dos escalas en un eje), colores de la paleta categórica validada en orden fijo (shared azul, local naranja, MPI aqua: el color sigue a la versión en todas las figuras), leyenda y nombre al final de cada línea, separados para que no se pisen (`etiquetar_sin_choque`, `common/plot.py:65`). Línea punteada vertical en los núcleos físicos y diagonal del speedup ideal.
- **Karp-Flatt** `e = (1/S − 1/p) / (1 − 1/p)` (`common/plot.py:48`), solo para p > 1.
- **`lscpu`**: en Windows no existe; se guarda lo equivalente de `wmic` (2 núcleos, 4 lógicos, 2.4 GHz, L3 3 MB).

**Qué tendría que poder explicar el humano en la sustentación:**
- Speedup, eficiencia y Karp-Flatt, y por qué aquí Karp-Flatt dice más del hardware que del código.
- Por qué con 8 trabajadores no mejora (4 hilos de hardware) y por qué con N = 200 000 escala mejor que con 20 000.
- No hay concurrencia en estas herramientas: `bench.py` corre una versión a la vez con `subprocess.run` y espera a que termine (las mediciones no se pisan entre sí).

**Dudas abiertas:**
- Las mediciones son de un portátil de 2 núcleos; en el computador de la demo los números cambiarán. Se recomienda correr `bench.py` allá.
- Con N = 200 000 hay una sola repetición por punto.
