# Mediciones comunes

<!-- PENDIENTE: se completa con results/demo_pc/ (Samuel en el PC de la demo) + results/omp/ + results/cuda/.
     Extensión objetivo: 1.5 páginas con 1 figura conjunta. Las tablas por versión quedan en su sección. -->

**Máquina y método.** Las versiones de CPU se midieron en el computador de la demo: Intel
i7-11800H, 8 núcleos físicos y 16 hilos, WSL2 Ubuntu (`results/omp/lscpu_demo.txt`). CUDA, en
Colab con una Tesla T4. `uniform`, θ = 0.5, 20 pasos, promedio de 3 repeticiones con N = 20 000 y
una con N = 200 000. T1 es el secuencial del mismo lenguaje: `py_core/` para multiprocessing y MPI,
`c_core/` para OpenMP. Todo se corrió desde el disco de WSL: leyendo desde `/mnt/c` la lectura de
200 000 partículas pasa de 0.41 s a 14.3 s y se vuelve la fracción secuencial dominante.

**Escalado con 1, 2, 4 y 8.** *(PENDIENTE: figura conjunta speedup y eficiencia de mp shared, mp
local, MPI y OpenMP para N = 20 000 y 200 000; tabla corta S(2), S(4), S(8) por versión.)*

**Más procesos o hilos que núcleos (16 y 32).** *(PENDIENTE: comparar las tres versiones de CPU.
OpenMP: de 8 a 16 sube poco, 5.42 → 6.67, por hyperthreading; con 32 baja a 6.18 por
sobresuscripción. Ver si multiprocessing y MPI siguen el mismo patrón y cuánto pesa crear 32
procesos.)*

**Desglose por fase en la GPU.** *(PENDIENTE: traer aquí la tabla morton/original de
`informe/sebastian_cuda.md` o dejarla allá y referenciarla, según el espacio.)*

**Por qué se aplanan las curvas.** *(PENDIENTE: redactar con los números de demo_pc.)*

- **Núcleos físicos frente a hilos virtuales**: hasta 8 escala casi lineal; los hilos 9–16
  comparten las unidades de un núcleo y solo ayudan mientras el otro hilo espera memoria.
- **Ancho de banda de memoria**: con N = 200 000 el árbol (≈ 69 000 nodos) no cabe en caché y los
  trabajadores compiten por la memoria al recorrerlo; por eso N grande escala peor en OpenMP.
- **Fracción secuencial**: el ordenamiento (`qsort` en C, el rango 0 en MPI), la lectura del
  archivo y, en Python, crear el `Pool` y cargar Numba en cada proceso. Karp-Flatt casi constante
  indica esa fracción (Amdahl); si crece con p indica sobrecosto (comunicación, sincronización,
  desbalance), como MPI con más rangos que núcleos.
- **Desbalance**: con `corner`, `static` en OpenMP y el reparto por cantidad en MPI dejan a un
  trabajador con hasta 2.8 veces más interacciones; el reparto dinámico (OpenMP `dynamic,64`,
  multiprocessing con `imap_unordered`) lo corrige.
