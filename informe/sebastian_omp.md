# Barnes-Hut con OpenMP (C)

**Diseño.** El núcleo secuencial `c_core/` ordena en cada paso las partículas por código de Morton
(21 bits por eje, `qsort`) y construye un octree **aplanado** en arreglos, con hojas de hasta 8
partículas. Ordenadas por Morton, las partículas de cada celda forman un tramo contiguo: el árbol se
arma partiendo rangos, sin insertar una por una. El recorrido usa una pila explícita y el integrador
es leapfrog kick-drift-kick; CUDA usa las mismas estructuras. `c_omp/bh_omp.c` paraleliza:

- **Fuerzas** (≈ 90% del tiempo): `#pragma omp for schedule(runtime)`, `dynamic,64` por defecto;
  `OMP_SCHEDULE` lo cambia sin recompilar. Cada `i` escribe solo su aceleración y el árbol solo se
  lee: no hay carreras.
- **Construcción sin candados**: tareas recursivas `omp task if(n > N/(8·hilos))`, cortadas por
  **cantidad de partículas** y no por nivel (con `corner`, el 95% de las partículas cae en una sola
  celda de nivel 2). Cada tarea escribe en su casilla de `hijos`; lo único compartido es el contador
  de nodos (`omp atomic capture`). La memoria se reserva antes con un conteo de nodos, para no hacer
  `realloc` entre hilos, y `taskwait` asegura que la masa de un nodo se calcule después que la de
  sus hijos.
- **Caja** con `reduction(min/max)`; Morton e integración con `parallel for`. Quedan secuenciales el
  `qsort` y el conteo de nodos.

*(Escalado: ver Mediciones comunes.)*

**Balanceo: `static` frente a `dynamic`** (N = 20 000, 100 pasos, `results/omp/*_100p_*.csv`):

| Distribución | Hilos | static | dynamic,64 | Interacciones máx/mín con static |
| --- | --- | --- | --- | --- |
| uniform | 8 | 4.10 s | 3.47 s | 1.05 |
| corner | 4 | 8.63 s | 7.43 s | 2.02 |
| corner | 8 | 6.73 s | 4.64 s | 2.76 |

Con `corner` y `static`, el hilo al que le toca el cúmulo hace 2.8 veces más interacciones que el
menos cargado y los hilos terminan entre 1.5 y 5.6 s. Con `dynamic,64` la razón baja a 1.09, todos
terminan a los 4.17 s y el total baja 31%. Con `uniform` las partículas cuestan casi lo mismo.

**La E/S también es Amdahl.** Leyendo desde `/mnt/c` (OneDrive visto desde WSL), las 200 000
partículas tardan 14.3 s en lugar de 0.41 s, y el speedup con 16 hilos cae a 2.2
(`results/omp/*_mntc_*`). Las mediciones y la demo corren desde el disco de WSL.

**Fracción secuencial.** Con 16 hilos y N = 200 000, el `qsort` (0.45 s) y la lectura (0.41 s)
son ≈ 11% del total, cerca del 0.12 de Karp-Flatt.
