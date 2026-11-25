# GUIA — Barnes-Hut con OpenMP (`c_omp/bh_omp.c`)

Una página para estudiar la versión sin reunirse. Detalle y números en `c_omp/NOTAS.md`; el núcleo secuencial en `c_core/NOTAS.md`.

```
make -C c_core && make -C c_omp                                  # en WSL o Linux
./c_omp/bh_omp --input data/corner_N20000_s42.txt --steps 20 --dt 2.5e-4 --workers 4 --out salida/omp
OMP_SCHEDULE="static" ./c_omp/bh_omp ...                          # cambia el reparto sin recompilar
```

## 1. Qué se paraleliza, por qué se puede y qué queda secuencial

- **Fuerzas** (`c_omp/bh_omp.c:172`), el ≈ 90% del tiempo: `#pragma omp for schedule(runtime)`. Se puede porque cada partícula `i` solo **lee** el árbol y **escribe** su propia `ax[i], ay[i], az[i]`.
- **Árbol** (`:130`): con las partículas ordenadas por Morton, cada subárbol es un tramo contiguo; un subárbol con más de N / (8 · hilos) partículas se vuelve una `task`. Se corta por **cantidad** y no por nivel: con `corner` el 95% de las partículas cae en una sola celda de nivel 2.
- **Caja** (`:60`, `reduction(min/max)`), **códigos Morton** (`:79`) e **integración** (`:182`, `:190`): `parallel for`, cada partícula es independiente.
- **Secuencial**: el `qsort` de los códigos (`:202`, ≈ 6% del tiempo con `corner`), el conteo previo de nodos (`:152`) y la lectura/escritura de archivos. Por Amdahl, con 6% secuencial el speedup no pasa de ≈ 17 ni con infinitos hilos.

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
| reparto de fuerzas | `OMP_SCHEDULE="static"`, `"dynamic,64"`, `"guided"`; defecto en `c_omp/bh_omp.c:25-26` | `static` con `corner`: un hilo hace 1.46× más interacciones y el total sube de 2.30 a 3.20 s |
| hilos | `--workers K` (`:44`) | más hilos que núcleos = sobresuscripción |
| corte de tasks | `FACTOR_CORTE` (`:30`), recompilar | más grande = más tasks, más pequeñas |
| θ | `--theta` (defecto 0.5 en `c_core/bh_io.c`, `leer_parametros`) | θ = 1.0: menos celdas abiertas, más rápido y menos preciso; no explota (la celda propia se abre siempre, `c_core/bh_fuerzas.c:49`) |
| hojas, bits Morton | `HOJA_MAX`, `BITS_EJE` en `c_core/bh_core.h` | tamaño del árbol |
| dt, ε | `--dt`, `--eps` | con `corner`/`clusters` usar dt = 2.5e-4 |

## 4. Tres preguntas de predicción

1. **¿Qué pasa de 4 a 8 hilos en una máquina de 4 núcleos?** El tiempo se queda casi igual o empeora un poco: los 8 hilos comparten 4 núcleos y se suma el costo de crear tasks y repartir bloques. Con hyperthreading puede mejorar algo (el recorrido espera memoria), nunca al doble.
2. **Con `corner`, ¿`static` o `dynamic`?** `dynamic`: con `static` al hilo que le toca el cúmulo le tocan las partículas que más celdas abren (353 contra 241 millones de interacciones) y los demás lo esperan. Con `uniform` casi empatan, porque todas las partículas trabajan parecido.
3. **¿Por qué el speedup no llega a 4 con 4 hilos?** El `qsort` es secuencial (Amdahl), crear tasks cuesta, y los hilos compiten por el ancho de banda de memoria al leer el árbol. Medido: 3.2 con `corner`.

## 5. Qué se ve en el monitor de recursos

*Lo esperado; se confirma con el monitor abierto en C4, en el computador de la demo.*
- **`uniform`, `dynamic`**: los K núcleos casi al 100% durante el recorrido, con bajones cortos en cada paso cuando un solo hilo hace el `qsort`.
- **`corner`, `static`**: al final de cada paso un solo núcleo sigue al 100% y los demás bajan: es el hilo con el cúmulo. En `PREFIJO_workers.csv` se ve como un hilo con más segundos e interacciones.
- **`corner`, `dynamic`**: vuelve a verse parejo; los hilos terminan juntos.
- **8 hilos en 4 núcleos**: el uso total no pasa de 400%; el sistema reparte el tiempo entre hilos.
