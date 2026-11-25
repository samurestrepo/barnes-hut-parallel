# Mapa de líneas clave para la sustentación (versiones de Samuel)

Solo el mapa: archivo:línea y una frase de qué hace. Orden sugerido de estudio: S2 → S3 → S4. Los detalles y el porqué están en `py_core/NOTAS.md`, `py_mp/NOTAS.md`, `mpi/NOTAS.md` y en las GUIA.

## Núcleo secuencial — `py_core/bh_core.py`

| Línea | Qué hace |
| --- | --- |
| `py_core/bh_core.py:29` | `THETA = 0.5`: criterio de apertura por defecto (perilla; también `--theta`). |
| `py_core/bh_core.py:32` | `HOJA_MAX = 8`: hasta 8 partículas por hoja (contrato). |
| `py_core/bh_core.py:36` | `BITS_MORTON = 21`: 21 bits por eje, 63 bits caben en un int64. |
| `py_core/bh_core.py:39` | `TAM_PILA = 256`: la pila nunca pasa de 7 × 21 + 8 = 155. |
| `py_core/bh_core.py:47` | `Arbol`: el octree en 9 arreglos planos; los hijos de un nodo son contiguos. |
| `py_core/bh_core.py:53` | Cubo raíz desde los extremos: se recalcula cada paso; MPI le pasa los globales. |
| `py_core/bh_core.py:74` | Recorta la coordenada cuantizada a 2^21 − 1 (la partícula del borde superior). |
| `py_core/bh_core.py:78` | Código de Morton intercalando bits x, y, z con un bucle. |
| `py_core/bh_core.py:107` | `argsort` estable: `orden[k]` es el índice original de la k-ésima partícula ordenada. |
| `py_core/bh_core.py:128` | Cada hijo es un tramo contiguo de los códigos ordenados. |
| `py_core/bh_core.py:146` | Centro del hijo: ± mitad según el bit del octante (x es el bit alto). |
| `py_core/bh_core.py:170` | Construcción con pila explícita, sin recursión. |
| `py_core/bh_core.py:188` | Hoja si hay ≤ 8 partículas o si se acabaron los bits (códigos repetidos). |
| `py_core/bh_core.py:196` | Capacidad 2N + 64 nodos; si no cabe, duplica y repite. |
| `py_core/bh_core.py:213` | Masas de abajo hacia arriba recorriendo los nodos al revés (los hijos tienen índice mayor). |
| `py_core/bh_core.py:235` | Término de gravedad del contrato: G m (r_j − r_i) / (r² + ε²)^{3/2}. |
| `py_core/bh_core.py:244` | Criterio θ; si la partícula está dentro de la caja de la celda, se abre siempre. |
| `py_core/bh_core.py:259` | `2·mitad < θ·d`: lado/d < θ sin dividir. |
| `py_core/bh_core.py:263` | Recorrido con pila explícita: aceptar → cm; hoja → suma directa; si no → apilar hijos. |
| `py_core/bh_core.py:284` | `j != k`: una partícula no se suma a sí misma. |
| `py_core/bh_core.py:296` | Aceleraciones de una lista de partículas (lo usa MPI). |
| `py_core/bh_core.py:308` | Aceleraciones de un rango [ini, fin) (el trozo de un trabajador de S3). |
| `py_core/bh_core.py:315` | Medio kick: v += a·dt/2. |
| `py_core/bh_core.py:320` | Drift: x += v·dt. |
| `py_core/bh_core.py:351` | `acc[orden] = acc_ord`: vuelta al orden de la entrada. |
| `py_core/bh_core.py:358` | Calentamiento de Numba que no se mide. |
| `py_core/bh_core.py:381` | Leapfrog KDK: kick, drift, fuerzas nuevas, kick. |

## multiprocessing — `py_mp/bh_mp.py`

| Línea | Qué hace |
| --- | --- |
| `py_mp/bh_mp.py:23` | `NUMBA_NUM_THREADS=1` antes de importar Numba: un hilo por proceso. |
| `py_mp/bh_mp.py:40` | `TROZOS_POR_PROCESO = 6`: perilla del balanceo. |
| `py_mp/bh_mp.py:49` | Crea un bloque de `shared_memory` con un arreglo de NumPy encima. |
| `py_mp/bh_mp.py:59` | El trabajador abre el bloque con `track=False`: el dueño es el principal. |
| `py_mp/bh_mp.py:70` | Liberar = `close` + `unlink` de cada bloque. |
| `py_mp/bh_mp.py:78` | Posiciones, masas y aceleraciones en memoria compartida, en el orden de la entrada. |
| `py_mp/bh_mp.py:98` | Modo shared: el principal construye el árbol y lo copia a memoria compartida. |
| `py_mp/bh_mp.py:120` | Trozos contiguos en orden Morton. |
| `py_mp/bh_mp.py:136` | `PARALELO`: `imap_unordered` reparte los trozos al primer proceso libre. |
| `py_mp/bh_mp.py:140` | `SINCRONIZACIÓN`: el `for` es la barrera; no se integra hasta tener todos los trozos. |
| `py_mp/bh_mp.py:143` | `SUPOSICIÓN`: cómo se reparte el tiempo entre construcción y recorrido en modo local. |
| `py_mp/bh_mp.py:161` | Inicializador del trabajador: abre la memoria compartida y carga Numba de la caché. |
| `py_mp/bh_mp.py:172` | Vistas sin copia del árbol compartido; reabre si el principal recreó los bloques. |
| `py_mp/bh_mp.py:185` | Modo local: un árbol por proceso y por paso. |
| `py_mp/bh_mp.py:189` | `SINCRONIZACIÓN`: el trabajador lee `pos` cuando el principal no la escribe. |
| `py_mp/bh_mp.py:212` | La tarea de un trozo: obtiene el árbol, calcula y escribe sus filas de `acc`. |
| `py_mp/bh_mp.py:222` | `SINCRONIZACIÓN`: filas disjuntas de `acc`, sin candado. |
| `py_mp/bh_mp.py:243` | Leapfrog en el principal sobre la memoria compartida. |
| `py_mp/bh_mp.py:276` | El `Pool` se crea una sola vez. |
| `py_mp/bh_mp.py:285` | `finally`: libera la memoria compartida después de cerrar el Pool, también si hubo error. |
| `py_mp/bh_mp.py:299` | Compila Numba en el principal antes de crear el Pool. |

## MPI — `mpi/bh_mpi.py`

| Línea | Qué hace |
| --- | --- |
| `mpi/bh_mpi.py:29` | `MPICH_PROGRESS_SPIN_LIMIT=16` antes de iniciar MPI (perilla de MS-MPI). |
| `mpi/bh_mpi.py:47` | `PARTICION = "cantidad"`: perilla (también `--particion`). |
| `mpi/bh_mpi.py:51` | Cada partícula viaja como 8 floats, con su índice original. |
| `mpi/bh_mpi.py:63` | Distancia de un punto al punto más cercano de una caja. |
| `mpi/bh_mpi.py:75` | Si la celda toca la caja remota, se abre siempre. |
| `mpi/bh_mpi.py:87` | Criterio θ con la distancia mínima: vale para todas las partículas de la caja. |
| `mpi/bh_mpi.py:97` | Poda: celda lejana → (cm, masa); hoja cercana → sus partículas; si no → abrir. |
| `mpi/bh_mpi.py:134` | Colectivas 1–2: `Allreduce` de mínimos y máximos (caja global). |
| `mpi/bh_mpi.py:141` | Cortes por rango: cantidad = N/P; espacio = tramos iguales del espacio de códigos. |
| `mpi/bh_mpi.py:152` | Colectivas 3–5: `Gather` y `Gatherv` de partículas y códigos al rango 0. |
| `mpi/bh_mpi.py:172` | La parte secuencial: el rango 0 ordena por Morton. |
| `mpi/bh_mpi.py:179` | Colectivas 6–8: `Scatter` y `Scatterv`; los demás esperan aquí mientras el 0 ordena. |
| `mpi/bh_mpi.py:188` | Un reparto completo: caja global, Morton, juntar en el 0, repartir. |
| `mpi/bh_mpi.py:210` | Colectiva 9: `Allgather` de cajas, partículas y nodos de cada rango. |
| `mpi/bh_mpi.py:215` | Arma un solo búfer con lo que va a cada rango remoto. |
| `mpi/bh_mpi.py:233` | Colectiva 10: `Alltoall` de cuentas, para reservar el búfer de recepción. |
| `mpi/bh_mpi.py:237` | Colectiva 11: `Alltoallv` de las ramas podadas (sin Send/Recv). |
| `mpi/bh_mpi.py:242` | Bytes del árbol completo: (nodos + partículas) × 32 de los otros rangos. |
| `mpi/bh_mpi.py:250` | Un paso de fuerzas por rango: árbol local, poda, intercambio, aceleraciones. |
| `mpi/bh_mpi.py:271` | Árbol con lo propio + lo recibido; recorrido solo para las propias. |
| `mpi/bh_mpi.py:281` | Dónde quedó cada partícula propia en el árbol combinado. |
| `mpi/bh_mpi.py:304` | Al final, todo vuelve al orden de la entrada en el rango 0. |
| `mpi/bh_mpi.py:310` | Leapfrog: kick, drift, nuevo reparto (con la velocidad), fuerzas, kick. |
| `mpi/bh_mpi.py:330` | Calentamiento: el 0 compila primero; barreras para que todos salgan juntos. |
| `mpi/bh_mpi.py:383` | `times.csv`: por fase, el tiempo del rango más lento. |

## Herramientas comunes

| Línea | Qué hace |
| --- | --- |
| `common/verify.py:39` | Las 5 pruebas del contrato con sus umbrales. |
| `common/bench.py:32` | Trabajadores por defecto: potencias de 2 hasta 2× los núcleos lógicos. |
| `common/bench.py:66` | `--lanzador` con `{k}`: `mpiexec -n {k}` o `mpirun --oversubscribe -n {k}`. |
| `common/plot.py:48` | Karp-Flatt: e = (1/S − 1/p) / (1 − 1/p). |
