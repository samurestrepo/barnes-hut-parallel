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
