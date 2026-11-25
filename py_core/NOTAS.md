# Notas de `py_core/` (S2)

Bitácora para estudiar en otro chat. Las entradas nuevas van al final.

---

## S2 — Barnes-Hut secuencial en Python (2026-11-24)

**Archivos creados:**
- `py_core/bh_core.py`: Barnes-Hut secuencial con Numba y el CLI del contrato. Es también la biblioteca que importan `py_mp/` (S3) y `mpi/` (S4).
- `results/py_core/uniform_N20000_r{1,2,3}_times.csv`, `resumen_uniform_N20000.csv`: tiempos con `uniform`, N = 20 000, 20 pasos.
- `results/py_core/deriva_energia.csv`: todas las derivas medidas abajo.

**Archivos cambiados:** `CONTRATO_DUDAS.md` (propuesta de umbral para `energia_movimiento`).

**Resultado del verificador** (`python common/verify.py correr --cmd "python py_core/bh_core.py"`):

| Prueba | Veredicto | Valor | Umbral |
| --- | --- | --- | --- |
| fuerzas_directa | PASA | 1.49e-14 | 1e-6 |
| fuerzas_bh | PASA | 3.25e-3 | 1e-2 |
| trayectoria | PASA | 2.78e-17 | 1e-6 |
| energia_reposo | PASA | 3.37e-8 | 1e-5 |
| energia_movimiento | **FALLA** | 8.87e-3 | 1e-3 (provisional) |

`energia_movimiento` falla por el umbral provisional, no por un bug. Pruebas que lo sostienen (`results/py_core/deriva_energia.csv`):
- Con θ = 0 la misma corrida da 3.98e-4, igual que la suma directa (4.0e-4). El integrador y el orden de salida están bien (además, `trayectoria` da 2.8e-17).
- La deriva crece de forma suave con θ: 3.7e-4 (0.1), 1.9e-3 (0.3), 8.9e-3 (0.5), 1.9e-2 (0.7). Es el error de la aproximación.
- A tiempo físico fijo (t = 0.025) **no baja con dt**: 7.6e-3 (dt = 5e-4), 8.9e-3 (2.5e-4), 9.2e-3 (1.25e-4). Si fuera el integrador, bajaría como dt². Viene del error de fuerza de θ = 0.5.
- Crece mucho al final: 1.6e-5 a los 10 pasos, 4.6e-4 a los 50, 8.9e-3 a los 100. Los cúmulos están fríos (dispersión 0.3 frente a velocidad virial ≈ 3) y colapsan en ≈ 0.016 unidades de tiempo; en el núcleo colapsado las aceleraciones son grandes y el error de θ pesa más.
- El error de fuerza con θ = 0.5 sobre esos mismos datos es el normal de Barnes-Hut de monopolo: mediana 3.6e-3, percentil 90 6.7e-3.
- Propuesta en `CONTRATO_DUDAS.md`: umbral 2e-2 (2.3× lo medido con θ = 0.5; el leapfrog sin segundo medio kick da 2.0e-1 y sigue fallando). No se cambió `verify.py` ni `CONTRATO.md`.

**Otras pruebas pedidas:**
- θ = 1.0 con `corner` N = 2000, dt = 2.5e-4, 100 pasos: corre sin NaN ni infinitos, deriva 9.19e-3 (con θ = 0.5: 8.59e-4).
- Posiciones fuera de [0, 1]: `gen.py --escala 10` con `uniform` (deriva 6.4e-11 en 10 pasos) y con `corner` (3.5e-8). La caja se recalcula de las posiciones, así que la escala no importa.
- `NUMBA_DISABLE_JIT=1` con N = 200 y 2 pasos: corre (1.8 s) y da exactamente las mismas salidas que con JIT (diferencia máxima 0.0).
- Tiempo con `uniform` N = 20 000, θ = 0.5, 20 pasos (promedio de 3 corridas, `results/py_core/resumen_uniform_N20000.csv`): **0.161 s por paso** (simulación 3.22 s, sin la lectura). Desglose: recorrido 94.8%, ordenamiento 2.9%, construcción 1.7%, integración 0.2%. El tiempo por paso divide entre 20, aunque hay 21 cálculos de fuerza (el del paso 0 también se mide).

**Decisiones y por qué:**
- **Árbol en arreglos planos** (`Arbol`, `py_core/bh_core.py:47`): cada nodo es una posición en 9 arreglos (rango de partículas, primer hijo, número de hijos, nivel, centro, mitad del lado, masa, centro de masa). Los hijos de un nodo son contiguos. Así Numba lo compila, y S3 lo puede poner en memoria compartida arreglo por arreglo.
- **Las partículas del nodo son un rango [ini, fin) del arreglo ordenado por Morton**: como el orden Morton agrupa por celda, cada hijo es un tramo contiguo y la construcción solo recorre los códigos ordenados (`fin_del_octante`, `:128`). No hay punteros por partícula.
- **Construcción con pila explícita** (`llenar_arbol`, `:170`) y capacidad inicial 2N + 64 nodos que se duplica si no alcanza (`construir_arbol`, `:196`). Evita reservar la cota del peor caso (≈ 19N nodos).
- **Hoja si hay ≤ 8 partículas o si se llegó al nivel 21** (`:188`): Morton de 21 bits puede repetir códigos y sin ese corte el árbol bajaría para siempre.
- **Morton con un bucle de 21 iteraciones** (`intercalar_bits`, `:78`) en vez de los "números mágicos": se lee qué hace. x es el bit alto de cada trío, igual que en `crear_hijo` (`:146`), así el octante del código y el centro del hijo coinciden.
- **Caja raíz cúbica** desde el mínimo y el lado mayor (`cubo_desde_extremos`, `:53`), recalculada en cada paso. Separada de `caja_raiz` para que MPI le pase los extremos globales del `allreduce`.
- **Masas de abajo hacia arriba recorriendo los nodos al revés** (`calcular_masas`, `:213`): un hijo siempre tiene índice mayor que su padre.
- **Criterio θ** (`acepta_celda`, `:244`): `2·mitad < θ·d` es lado/d < θ sin dividir. Antes, si la partícula está dentro de la caja de la celda (con tolerancia relativa 1e-9, `TOL_CAJA`, `:42`, por el redondeo de partir 21 veces), la celda se abre siempre. Así θ = 1 no suma a la partícula consigo misma.
- **Recorrido con pila de tamaño fijo** (`aceleracion_de`, `:263`; `TAM_PILA = 256`, `:39`): cada nivel deja a lo sumo 7 hermanos pendientes, 7·21 + 8 = 155 < 256.
- **Funciones reutilizables**: `aceleraciones_rango(ini, fin, …)` (`:308`) es el trozo de un trabajador de S3; `aceleraciones_lista(objetivos, …)` (`:296`) es lo que usa MPI para calcular solo sus partículas dentro de un árbol con partículas ajenas.
- **Fases medidas**: `ordenamiento` = caja + Morton + argsort + copias en orden; `construccion` = árbol + masas; `recorrido` = fuerzas + desordenar; `integracion` = kicks y drift; `comunicacion` = 0.
- **Calentamiento no medido** (`calentar_numba`, `:358`, llamado en `:403`) con 64 partículas: Numba compila por tipos, no por tamaño. Se comprobó que cada función tiene una sola firma (no se recompila dentro de la medición) y que, con la caché (`cache=True`), el calentamiento tarda 0.6 s.

**Qué tendría que poder explicar el humano en la sustentación:**
- El ciclo de un paso: medio kick, drift, árbol nuevo (las partículas se movieron), fuerzas, medio kick (`simular`, `:381`; kicks en `:387` y `:392`).
- Por qué hay que desordenar: `acc[orden] = acc_ord` (`:351`). `orden[k]` es el índice original de la partícula que quedó en la posición k.
- Por qué `cuantizar` recorta a 2^21 − 1 (`:74`): la partícula del borde superior caería fuera de la malla.
- Por qué θ = 0 da la suma directa: ninguna celda cumple `2·mitad < 0`, así que se llega a todas las hojas y se suma partícula por partícula (`if j != k`, `:284`, evita la auto-atracción).
- Qué cuenta como interacción (para `workers.csv` de S3 y S4): un término de gravedad, sea una partícula de una hoja o una celda aceptada.
- No hay concurrencia en S2: ni carreras ni deadlocks. La única "trampa" de orden es la de desordenar las salidas.
- Cómo depurar sin IA: `NUMBA_DISABLE_JIT=1` y N = 200; los `@njit` se vuelven Python normal y se puede poner `print`.

**Dudas abiertas:**
- Umbral de `energia_movimiento`: se propone 2e-2 en `CONTRATO_DUDAS.md`. Mientras no se cambie, S2, S3 y S4 marcan FALLA en esa prueba con el mismo valor (≈ 8.9e-3).
- `HOJA_MAX` y `TOL_CAJA` no son perillas de la demo; se dejan arriba del archivo por si se quieren probar.
