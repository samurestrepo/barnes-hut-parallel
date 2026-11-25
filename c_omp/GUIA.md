# GUIA — Barnes-Hut con OpenMP (`c_omp/bh_omp.c`)

Una página para estudiar la versión sin reunirse. Detalle y números en `c_omp/NOTAS.md`; el núcleo secuencial en `c_core/NOTAS.md`.

```
make -C c_core && make -C c_omp                                  # en WSL, desde ~/ (no /mnt/c: leer es 35 veces más lento)
./c_omp/bh_omp --input data/corner_N20000_s42.txt --steps 20 --dt 2.5e-4 --workers 4 --out salida/omp
OMP_SCHEDULE="static" ./c_omp/bh_omp ...                          # cambia el reparto sin recompilar
```

## 1. Qué se paraleliza, por qué se puede y qué queda secuencial

- **Fuerzas** (`c_omp/bh_omp.c:172`), el ≈ 90% del tiempo: `#pragma omp for schedule(runtime)`. Se puede porque cada partícula `i` solo **lee** el árbol y **escribe** su propia `ax[i], ay[i], az[i]`.
- **Árbol** (`:130`): con las partículas ordenadas por Morton, cada subárbol es un tramo contiguo; un subárbol con más de N / (8 · hilos) partículas se vuelve una `task`. Se corta por **cantidad** y no por nivel: con `corner` el 95% de las partículas cae en una sola celda de nivel 2.
- **Caja** (`:60`, `reduction(min/max)`), **códigos Morton** (`:79`) e **integración** (`:182`, `:190`): `parallel for`, cada partícula es independiente.
- **Secuencial**: el `qsort` de los códigos (`:202`), el conteo previo de nodos (`:152`) y la lectura/escritura de archivos. Con N = 200 000 el `qsort` (0.45 s) y la lectura (0.41 s) son ≈ 2% del secuencial (45.5 s), pero con 16 hilos ya son ≈ 11% del total: lo paralelo se encoge y lo secuencial no (Amdahl).

## 2. Carreras y deadlocks: dónde podrían estar y cómo se evitaron

- **Contador de nodos** (`:104`): varias tasks crean nodos a la vez. `#pragma omp atomic capture` lee y aumenta `n_nodos` en una sola operación; sin él, dos tasks tomarían el mismo índice y una pisaría el nodo de la otra.
- **`realloc` dentro de las tasks**: evitado. Antes de repartir se cuentan los nodos y se reserva todo (`:152`); si un hilo hiciera `realloc`, otro seguiría escribiendo en la memoria vieja.
- **Masa de un nodo antes de que terminen sus hijos**: `#pragma omp taskwait` (`:134`) antes de `resumir_interno`.
- **Integrar con fuerzas a medio calcular**: la barrera implícita al cerrar el `parallel` de fuerzas (`:177`). El `nowait` del `for` (`:172`) solo quita la barrera interna para que cada hilo mida cuándo terminó SU parte.
- **Contadores por hilo** (`segundos`, `interacciones`, `construidas`): cada hilo escribe solo la casilla `[omp_get_thread_num()]`, sin carrera.
- **Deadlocks**: no hay candados ni esperas cruzadas, así que no puede haberlos. Prueba de que no hay carreras: las salidas con 1 y con 8 hilos son idénticas byte a byte.

## 3. Perillas de la demo

| Perilla | Dónde | Qué cambia |
| --- | --- | --- |
| reparto de fuerzas | `OMP_SCHEDULE="static"`, `"dynamic,64"`, `"guided"`; defecto en `c_omp/bh_omp.c:25-26` | `static` con `corner` (8 hilos, 100 pasos): un hilo hace 2.8× más interacciones y el total sube de 4.64 a 6.73 s |
| hilos | `--workers K` (`:44`) | más hilos que núcleos = sobresuscripción |
| corte de tasks | `FACTOR_CORTE` (`:30`), recompilar | más grande = más tasks, más pequeñas |
| θ | `--theta` (defecto 0.5 en `c_core/bh_io.c`, `leer_parametros`) | θ = 1.0: menos celdas abiertas, más rápido y menos preciso; no explota (la celda propia se abre siempre, `c_core/bh_fuerzas.c:49`) |
| hojas, bits Morton | `HOJA_MAX`, `BITS_EJE` en `c_core/bh_core.h` | tamaño del árbol |
| dt, ε | `--dt`, `--eps` | con `corner`/`clusters` usar dt = 2.5e-4 |

## 4. Tres preguntas de predicción

1. **¿Qué pasa de 8 a 16 y a 32 hilos en el PC de la demo (8 núcleos, 16 hilos)?** De 8 a 16 mejora poco (speedup 5.42 → 6.67 con N = 20 000): los hilos extra son hyperthreading y comparten núcleo; ayudan algo porque el recorrido espera memoria. Con 32 empeora un poco (6.18): hay más hilos que CPUs lógicas y el sistema los turna.
2. **Con `corner`, ¿`static` o `dynamic`?** `dynamic`: con `static` al hilo que le toca el cúmulo le tocan las partículas que más celdas abren (2.8 veces más interacciones que el menos cargado con 8 hilos) y los demás lo esperan. Con `uniform` casi empatan, porque todas las partículas trabajan parecido.
3. **¿Por qué el speedup no llega a 8 con 8 hilos?** El `qsort` y la lectura son secuenciales (Amdahl), crear tasks cuesta, y los hilos compiten por el ancho de banda de memoria al leer el árbol. Medido: 5.42 con N = 20 000 y 4.50 con N = 200 000 (el árbol grande no cabe en caché).

## 5. Qué se ve en el monitor de recursos

*Lo esperado en el PC de la demo (8 núcleos / 16 hilos); vale la pena mirarlo una vez con el Administrador de tareas antes del ensayo.*
- **`uniform`, `dynamic`**: los K núcleos casi al 100% durante el recorrido, con bajones cortos en cada paso cuando un solo hilo hace el `qsort`.
- **`corner`, `static`**: al final de cada paso un solo núcleo sigue al 100% y los demás bajan: es el hilo con el cúmulo. En `PREFIJO_workers.csv` se ve como un hilo con más segundos e interacciones.
- **`corner`, `dynamic`**: vuelve a verse parejo; los hilos terminan juntos.
- **32 hilos en 16 CPUs lógicas**: el uso no pasa del 100% de la CPU; el sistema turna los hilos y el tiempo empeora un poco (sobresuscripción).
