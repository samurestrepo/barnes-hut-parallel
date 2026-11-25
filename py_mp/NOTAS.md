# Notas de `py_mp/` (S3)

Bitácora para estudiar en otro chat. Las entradas nuevas van al final.

---

## S3 — multiprocessing, árbol compartido vs local (2026-11-24)

**Archivos creados:**
- `py_mp/bh_mp.py`: Barnes-Hut con `multiprocessing.Pool`, modos `--tree shared` y `--tree local`. Importa el árbol y las fórmulas de `py_core/bh_core.py` (S2) sin copiar código.
- `results/mp/uniform_N20000_{shared,local}_w{1,2,4,8}_r{1,2}_{times,workers}.csv`: 2 corridas por configuración.
- `results/mp/resumen_uniform_N20000.csv`: promedio, tiempo por paso, speedup y eficiencia contra S2.
- `results/mp/corner_N20000_{shared,local}_w4_{times,workers}.csv`: balanceo con `corner`.

**Resultado del verificador** (`python common/verify.py correr --cmd "python py_mp/bh_mp.py --tree X" --workers K`): los 4 casos (shared/local × 1/4 trabajadores) dan **exactamente** los valores de S2:

| Prueba | shared w1 | shared w4 | local w1 | local w4 |
| --- | --- | --- | --- | --- |
| fuerzas_directa | PASA 1.49e-14 | PASA 1.49e-14 | PASA 1.49e-14 | PASA 1.49e-14 |
| fuerzas_bh | PASA 3.25e-3 | PASA 3.25e-3 | PASA 3.25e-3 | PASA 3.25e-3 |
| trayectoria | PASA 2.78e-17 | PASA 2.78e-17 | PASA 2.78e-17 | PASA 2.78e-17 |
| energia_reposo | PASA 3.37e-8 | PASA 3.37e-8 | PASA 3.37e-8 | PASA 3.37e-8 |
| energia_movimiento | FALLA 8.87e-3 | FALLA 8.87e-3 | FALLA 8.87e-3 | FALLA 8.87e-3 |

`energia_movimiento` falla por el umbral provisional 1e-3, igual que S2 (ver `py_core/NOTAS.md` y la propuesta de 2e-2 en `CONTRATO_DUDAS.md`). Que las salidas sean idénticas bit a bit a S2 es lo esperado: cada partícula recorre el mismo árbol en el mismo orden; solo cambia qué proceso la calcula.

**Tiempos** (`uniform` N = 20 000, θ = 0.5, 20 pasos, promedio de 2 corridas; S2 secuencial = 3.315 s total, 0.161 s por paso; `results/mp/resumen_uniform_N20000.csv`):

| Modo | p | total (s) | por paso (s) | speedup | eficiencia |
| --- | --- | --- | --- | --- | --- |
| shared | 1 | 3.835 | 0.187 | 0.86 | 0.86 |
| shared | 2 | 2.439 | 0.117 | 1.36 | 0.68 |
| shared | 4 | 2.042 | 0.092 | 1.62 | 0.41 |
| shared | 8 | 2.176 | 0.104 | 1.52 | 0.19 |
| local | 1 | 5.086 | 0.246 | 0.65 | 0.65 |
| local | 2 | 2.596 | 0.125 | 1.28 | 0.64 |
| local | 4 | 2.005 | 0.095 | 1.65 | 0.41 |
| local | 8 | 2.344 | 0.111 | 1.42 | 0.18 |

Crear el Pool (`crear_pool` en times.csv, fuera de `total`): shared 1.8 / 2.0 / 3.8 / 7.2 s y local 2.4 / 2.5 / 3.4 / 6.7 s con 1 / 2 / 4 / 8 procesos.

**Cuál modo gana y por qué:**
- **shared gana o empata en todos los casos**: con 1–2 procesos por bastante (3.84 vs 5.09 s; 2.44 vs 2.60 s), con 4 empatan (2.04 vs 2.01 s, dentro del ruido entre corridas) y con 8 gana otra vez (2.18 vs 2.34 s).
- **Compartir el árbol cuesta casi nada**: copiarlo a la memoria compartida (`comunicacion` en shared) suma 0.006 s en 21 pasos, unos 0.3 ms por paso. Las partículas ya viven en memoria compartida y los trabajadores leen el árbol sin copiarlo ni serializarlo.
- **local repite la construcción en cada proceso**: la construcción por trabajador sube de 0.23 s (1 proceso) a 0.31 s (4) y 0.41 s (8), porque los procesos compiten por los 2 núcleos físicos y por la memoria. Es trabajo redundante: con p procesos se construye p veces el mismo árbol.
- **Lo que limita a shared** es la parte secuencial del principal: ordenar y construir (0.17 s, el 8% del total con 4 procesos) mientras los trabajadores esperan. Es la fracción de Amdahl.

**Dónde está el overhead:**
1. **Crear el Pool** es lo más caro, de lejos: 1.8 a 7.2 s, más que toda la simulación con 4 procesos (2.0 s). En Windows los procesos se crean con *spawn*: cada uno arranca un intérprete nuevo, importa NumPy y Numba y carga la caché de Numba (`iniciar_trabajador`, `py_mp/bh_mp.py:161`). Por eso el Pool se crea **una sola vez** (`py_mp/bh_mp.py:276`) y no en cada paso: crearlo por paso costaría 20 × 2–7 s.
2. **Serializar**: casi nada. Por la cola de tareas solo viajan tuplas pequeñas (rango, paso, nombres de los bloques) y vuelven 4 números por trozo. Las posiciones, el árbol y las aceleraciones no viajan: están en memoria compartida.
3. **Copiar**: solo el árbol a la memoria compartida en modo shared, 0.3 ms por paso.
4. **Con 1 proceso, shared tarda un 16% más que S2** (recorrido 3.53 s contra 3.05 s): son los 6 trozos por paso repartidos por la cola, más la escritura con índices `acc[orden[ini:fin]]` y los hilos internos del Pool en el principal. Es el costo fijo de la maquinaria del Pool.
5. **La máquina tiene 2 núcleos físicos y 4 lógicos** (i5-6300U). De 2 a 4 procesos el speedup sube poco (1.36 → 1.62) porque el hyper-threading comparte las unidades de cálculo de un núcleo; de 4 a 8 baja (sobresuscripción: 8 procesos en 4 hilos de hardware, más cambios de contexto y más construcciones redundantes en local).

**Balanceo con `corner`** (N = 20 000, dt = 2.5e-4, 4 procesos, `results/mp/corner_N20000_*_w4_workers.csv`): los 4 trabajadores quedan dentro del 2% en segundos (shared: 1.87–1.90 s; local: 2.17–2.19 s) y en interacciones (60.5–62.2 millones), con ≈ 126 trozos cada uno. Con trozos de igual número de partículas, los del rincón denso abren más celdas y tardan más, pero `imap_unordered` le da el siguiente trozo al primero que se libera, así que nadie se queda esperando mucho.

**Decisiones y por qué:**
- **Posiciones, masas y aceleraciones en memoria compartida en el orden de la entrada** (`bloques_particulas`, `py_mp/bh_mp.py:78`). El principal integra sobre ellas en su sitio y los trabajadores escriben acc directamente; nada viaja por la cola.
- **Trozos contiguos en orden Morton** (`repartir_trozos`, `:120`): cada trozo es una región compacta del espacio, así que sus partículas recorren casi las mismas celdas (mejor caché).
- **`TROZOS_POR_PROCESO = 6`** (`:40`, perilla): entre 4 y 8 como pide el diseño. Más trozos balancean mejor y cuestan más tareas.
- **El árbol compartido cabe en bloques con capacidad 2N + 64 nodos**; si un paso necesita más, el principal crea bloques nuevos y los trabajadores los reabren porque cambian los nombres (`publicar_arbol`, `:98`; `arbol_compartido`, `:172`).
- **Modo local: un árbol por proceso y por paso** (`arbol_local`, `:185`), guardado con el número de paso para que los demás trozos del mismo paso lo reutilicen. Todos construyen el mismo árbol porque `argsort` estable es determinista.
- **SUPOSICIÓN** (`:143`): en modo local, `construccion` = promedio por trabajador de su construcción (corren en paralelo) y `recorrido` = el resto de la fase paralela.
- **Numba se compila en el principal antes de crear el Pool** (`:299`) y cada trabajador lo carga de la caché en el inicializador; `NUMBA_NUM_THREADS=1` (`:23`) va antes de importar Numba. `crear_pool` incluye esa carga y se espera a que todos los procesos la terminen (`esperar_trabajadores`, `:202`) para que no caiga dentro de la medición.

**Qué tendría que poder explicar el humano en la sustentación:**
- **No hay carrera en acc** (`:222`): cada trozo escribe filas distintas, porque `orden` es una permutación y los rangos [ini, fin) no se tocan. Sin candado.
- **La barrera es el `for` de `imap_unordered`** (`:140`): el principal no integra ni mueve pos hasta tener el resultado de TODOS los trozos. Si integrara antes, un trabajador atrasado leería posiciones del paso siguiente (carrera de lectura/escritura).
- **En local, los trabajadores leen pos mientras el principal espera** (`:189`); el principal solo escribe pos entre pasos, así que nadie la cambia durante la lectura.
- **Liberar la memoria compartida** (`:285`): `close` + `unlink` en un `finally`, después de que el `with Pool` terminó; si se liberara antes, un trabajador podría leer memoria ya liberada. Los trabajadores abren con `track=False` (`:59`) para que el rastreador de recursos no borre bloques que son del principal.
- **No puede haber deadlock**: no hay candados y la única espera es el principal esperando resultados de tareas que siempre terminan. Si un trabajador lanza una excepción, `imap_unordered` la vuelve a lanzar en el principal y el `finally` libera la memoria.
- Por qué el speedup se estanca en ≈ 1.6: 2 núcleos físicos + parte secuencial del principal + costo fijo del Pool.

**Dudas abiertas:**
- Umbral de `energia_movimiento` (igual que S2).
- En el computador de la demo, si es Linux/WSL, los procesos se crean con *fork* y `crear_pool` debería bajar mucho; conviene medirlo allá.
