# Diseño del proyecto Barnes-Hut — reparto, contrato y tareas

Nov 23, 2026 · @Samuel Restrepo

## Reparto y reglas

Cada uno toma una versión difícil y una normal, en un solo lenguaje, y trabaja solo en sus carpetas. Lo único común es el **contrato** de la sección siguiente: está fijado aquí para que nadie tenga que negociarlo.

| Persona | Difícil | Normal | Lenguaje | Entorno |
| --- | --- | --- | --- | --- |
| Samuel | MPI (mpi4py) | Python multiprocessing | Python + NumPy + Numba | WSL2 con OpenMPI |
| Sebastián | CUDA | C con OpenMP | C | gcc local + Colab T4 |

**Por qué este reparto:**

- Cada persona trabaja en un solo lenguaje y reutiliza su propio árbol en sus dos versiones: el árbol en Python sirve para multiprocessing y MPI; el árbol aplanado en C sirve para OpenMP y CUDA.
- Las dos versiones difíciles (MPI y CUDA) quedan en manos distintas, así un atasco no bloquea al otro.
- Samuel hace además las herramientas comunes (generador, verificador, mediciones) porque están en Python, como sus versiones.

**Reglas para no tener que coordinarse:**

1. Nadie toca las carpetas del otro. Hay una rama por implementación y cada una entra a `main` cuando pasa el verificador.
2. El contrato no se cambia sin avisar por chat. Si algo no funciona, se escribe en `CONTRATO_DUDAS.md` y se sigue.
3. Cada versión se da por terminada solo cuando pasa el verificador (`common/verify.py`).
4. Cada versión deja una `GUIA.md` de una página para que el otro la estudie sin reunirse.
5. Solo hay dos encuentros: un control corto el día 2 y el ensayo de la sustentación el día 5.

**Presupuesto:** ≈11 h de tareas + ≈2 h de encuentros por persona, entre el mié 25 y el lun 30. Es más que las 8–10 h disponibles: por eso cada versión difícil tiene un límite el viernes en la noche y una simplificación lista, y el informe (10%) es lo primero que se recorta. El código lo escribe la IA; las horas de cada uno van a supervisar, verificar y, sobre todo, estudiar: la sustentación y la demo valen el 90%.

## Contrato común

Las cuatro versiones leen el mismo archivo, usan la misma física, aceptan los mismos parámetros y escriben las mismas salidas. Con eso, un solo verificador sirve para todas y nadie depende del código del otro.

**Física (fija):**

| Parámetro | Valor | Nota |
| --- | --- | --- |
| G | 1 | del enunciado |
| ε (suavizado) | 0.01 | ajustable por CLI |
| dt | 0.001 con `uniform`; **2.5e-4 con distribuciones concentradas** (`corner`, `clusters`, la de la demo) | con `corner` y dt = 0.001 la energía no se conserva aunque el código esté bien (auditoría: deriva 2.7e-2; con 2.5e-4 baja a 2.0e-4) |
| θ | 0.5 por defecto | lado de la celda / distancia al centro de masa < θ → la celda cuenta como una partícula. **Si la partícula está dentro de la caja de la celda, la celda se abre siempre** (con θ ≳ 0.58 se sumaría a sí misma; la demo puede pedir θ = 1) |
| Hojas | hasta 8 partículas por hoja | Morton de 21 bits puede repetir códigos |
| Caja raíz | min/max de las posiciones, recalculada en cada paso | nunca suponer el cubo unitario: la demo puede traer posiciones fuera de \[0, 1\] |
| Integrador | leapfrog kick-drift-kick | medio paso de velocidad, paso completo de posición, recalcular aceleraciones, medio paso de velocidad |
| Pasos | 10 para verificar; 20 para medir (se reporta tiempo por paso); 100 si alcanza el tiempo | 20 en vez de 100: suposición a confirmar con el profesor |

Aceleración sobre la partícula i (se compara aceleración, no fuerza, para no depender de la masa de i):

```latex
a_i = \sum_{j \neq i} G\, m_j \, \frac{r_j - r_i}{\left(|r_j - r_i|^2 + \varepsilon^2\right)^{3/2}}
```

Energía total (la calcula solo el verificador):

```latex
E = \sum_i \tfrac{1}{2} m_i |v_i|^2 \; - \sum_{i<j} \frac{G\, m_i m_j}{\sqrt{|r_i - r_j|^2 + \varepsilon^2}}
```

**Entrada:** archivo de texto, N filas × 7 columnas separadas por espacio (`x y z vx vy vz m`), float64, sin encabezado. Nombre: `data/<dist>_N<N>_s<semilla>.txt`.

**Parámetros de línea de comandos (iguales en las 4 versiones):**

```
<programa> --input F --steps S --dt 0.001 --theta 0.5 --eps 0.01 --workers K --out PREFIJO
```

`--workers` = procesos (multiprocessing), rangos (MPI: lo da `mpirun -n`, el parámetro se ignora), hilos (OpenMP) o hilos por bloque (CUDA).

**Salidas obligatorias:**

| Archivo | Contenido | Para qué |
| --- | --- | --- |
| `PREFIJO_acc0.txt` | N×3 aceleraciones del paso 0, **en el orden de la entrada** | comparar contra la suma directa |
| `PREFIJO_final.txt` | N×7 estado tras S pasos, **en el orden de la entrada** | energía final |
| `PREFIJO_times.csv` | `fase,segundos` con `lectura, construccion, ordenamiento, recorrido, integracion, comunicacion, total` | informe y gráficas |

Ojo: casi todas las versiones reordenan las partículas por Morton. Hay que guardar el índice original y devolverlas a su orden antes de escribir. Si no, el verificador falla aunque la física esté bien.

**Correcta = pasa `common/verify.py` (criterios iniciales, a ajustar con mediciones):**

| Prueba | Datos | Criterio |
| --- | --- | --- |
| Fuerzas, suma directa | `uniform`, N = 2000, θ = 0 | error relativo máximo < 1e-6 (float64) o < 1e-3 (CUDA en float32) |
| Fuerzas, Barnes-Hut | `uniform`, N = 2000, θ = 0.5 | mediana del error relativo < 1e-2 |
| Trayectoria | `uniform`, N = 2000, θ = 0, 10 pasos | `verify.py` integra los mismos 10 pasos con suma directa y compara `final.txt` **posición por posición**: error máximo < 1e-6 (float64) o < 1e-3 (CUDA). Detecta un integrador mal hecho y un orden de salida equivocado |
| Energía, reposo | `uniform`, N = 2000, θ = 0, 100 pasos | deriva < 1e-5 (float64) o < 1e-3 (CUDA) |
| Energía, en movimiento | `clusters` con velocidades y masas aleatorias, N = 2000, θ = 0.5, dt = 2.5e-4, 100 pasos | deriva < 1e-3 (suposición: ajustarlo el mié 25 con S2) |

Por qué se endureció (auditoría): con la nube uniforme en reposo, 100 × 0.001 = 0.1 unidades de tiempo y el colapso tarda unas 0.5, así que las partículas casi no se mueven. Con el umbral anterior de 1e-2 pasaban un leapfrog sin el segundo medio kick (5.6e-3) y hasta Euler explícito (2.2e-4); el correcto da 3.4e-8. Además, la energía no depende del orden de las partículas, por eso ahora se compara la trayectoria. `verify.py` también imprime la deriva de cualquier archivo, incluida la distribución de la demo.

**Generador (cualquiera de los dos puede crearlo sin esperar al otro):**

```python
# common/gen.py  —  python common/gen.py --dist clusters --n 20000 --vel 0.3 --masas aleatorias
import argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument("--dist", default="uniform", choices=["uniform", "clusters", "corner"])
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--vel", type=float, default=0.0)       # 0 = reposo; >0 = velocidades aleatorias
p.add_argument("--masas", default="iguales", choices=["iguales", "aleatorias"])
p.add_argument("--escala", type=float, default=1.0)    # lado del cubo; prueba posiciones fuera de [0, 1]
a = p.parse_args()
rng = np.random.default_rng(a.seed)
if a.dist == "uniform":            # requisito del enunciado
    pos = rng.random((a.n, 3))
elif a.dist == "clusters":         # demo: dos cúmulos densos que chocan
    h = a.n // 2
    pos = np.vstack([rng.normal(0.3, 0.03, (h, 3)), rng.normal(0.7, 0.03, (a.n - h, 3))])
else:                              # demo: casi todo en un rincón
    k = int(a.n * 0.95)
    pos = np.vstack([rng.random((k, 3)) * 0.05, rng.random((a.n - k, 3))])
pos *= a.escala
vel = rng.normal(0.0, a.vel, (a.n, 3)) if a.vel > 0 else np.zeros((a.n, 3))
if a.dist == "clusters" and a.vel > 0:   # que se acerquen: uno a la derecha, otro a la izquierda
    vel[: a.n // 2, 0] += a.vel
    vel[a.n // 2 :, 0] -= a.vel
m = rng.random((a.n, 1)) if a.masas == "aleatorias" else np.ones((a.n, 1))
m /= m.sum()                       # masa total = 1
suf = ("_v" if a.vel > 0 else "") + ("_m" if a.masas == "aleatorias" else "") + ("_e" if a.escala != 1 else "")
np.savetxt(f"data/{a.dist}_N{a.n}_s{a.seed}{suf}.txt", np.hstack([pos, vel, m]))
```

Las distribuciones `clusters` y `corner` no las pide el enunciado: sirven para ensayar la demo, que usa una distribución desconocida "que castiga el reparto ingenuo".

## Estructura del repositorio y commits

Un solo repositorio en GitHub, una carpeta por implementación y **una rama por implementación** (decisión de Samuel y Sebastián, 24-nov). Cada rama entra a `main` cuando su versión pasa el verificador. Como cada rama solo toca su carpeta, los merges no tienen conflictos.

```
barnes-hut/
├── README.md            cómo correr cada versión (se une al final)
├── CLAUDE.md            reglas para los agentes
├── DISEÑO.md            este documento exportado
├── CONTRATO.md          copia de la sección "Contrato común"
├── CONTRATO_DUDAS.md    problemas con el contrato, sin bloquear al otro
├── common/              Samuel: gen.py, verify.py, bench.py, plot.py
├── data/                archivos generados (en .gitignore salvo uno pequeño de prueba)
├── py_core/             Samuel: árbol secuencial en Python (lo usan py_mp y mpi)
├── py_mp/               Samuel: multiprocessing + GUIA.md
├── mpi/                 Samuel: mpi4py + GUIA.md
├── c_core/              Sebastián: árbol aplanado secuencial en C (lo usan c_omp y cuda)
├── c_omp/               Sebastián: OpenMP + Makefile + GUIA.md
├── cuda/                Sebastián: notebook de Colab ejecutado + .cu + GUIA.md
├── results/             CSV de tiempos y gráficas (una subcarpeta por implementación)
└── informe/             una sección .md por implementación + una común
```

**Reglas de commits** (el historial cuenta en el informe):

- **Orden de las ramas:**
  1. `main` recibe directo la base común (S0 y S1) el miércoles 25.
  2. `impl/py-core` (S2) e `impl/c-core` (C1): el árbol secuencial de cada lenguaje. Entran a `main` apenas pasan el verificador, porque las dos implementaciones de cada persona dependen de ellas.
  3. `impl/mp` (S3), `impl/mpi` (S4), `impl/omp` (C2) e `impl/cuda` (C3), cada una creada desde `main` cuando ya tiene su núcleo.
  4. `tools/bench` (S5 y C4) y `docs/informe` (S6 y C5).
- **Entrar a `main`:** por pull request en GitHub y **sin squash**, para conservar el historial que pide el enunciado. El PR no espera aprobación del otro, pero su descripción lleva la salida del verificador y tres líneas de qué hace la versión. Leer los PR del otro es parte del estudio cruzado.
- **Si `main` cambia** (verificador o núcleo): `git merge main` en la rama de la implementación.
- Una persona puede tener dos ramas abiertas a la vez, por ejemplo `impl/mp` esperando mediciones mientras avanza `impl/mpi`.
- Las ramas se suben a GitHub desde el primer commit: si un computador falla, no se pierde nada.
- **Archivos que podrían chocar:** el `README.md` raíz y la parte común del informe. Se unen al final, en `docs/informe`.
- Un commit por tarea terminada, con el código de la tarea al inicio: `S3: multiprocessing con árbol en memoria compartida`. Nunca commitear código que no corre.
- Los archivos de datos grandes no se suben; se regeneran con `gen.py` y la semilla.
- El notebook de CUDA se sube **ejecutado**, con las salidas visibles, como pide el enunciado.

**Arreglos al núcleo:** si `impl/mpi` necesita cambiar `py_core/` (o `impl/cuda` cambiar `c_core/`), el arreglo no se hace en esa rama: va en `fix/py-core` o `fix/c-core` con PR a `main`, y después cada implementación hace `git merge main`. Así ninguna rama queda con un núcleo viejo. Resultados siempre en `results/<impl>/` (`mp`, `mpi`, `omp`, `cuda`).

## Tareas de Samuel (MPI + multiprocessing)

Son ocho tareas que suman ≈11 h, más ≈2 h de encuentros. S0 y S1 van primero porque Sebastián necesita el verificador desde el día 2. Cada tarea se le pasa al agente tal cual: "haz la tarea S2 de DISEÑO".

| Tarea | Rama | Qué se construye | Horas | Hecho cuando | Commit |
| --- | --- | --- | --- | --- | --- |
| S0 | `main` | Estructura del repo, `.gitignore`, `README.md`, `CLAUDE.md`, `DISEÑO.md`, `CONTRATO.md`, `common/gen.py` (código del contrato) | 0.5 | genera `uniform`, `clusters` y `corner` con N = 2000 y 20 000; Sebastián puede clonar | `S0: estructura y generador` |
| S1 | `main` | `common/verify.py`: suma directa N² vectorizada (por bloques), integración directa de 10 pasos, las 5 pruebas del contrato (fuerzas, trayectoria, energía) y la deriva de energía de cualquier archivo | 1.5 | con salidas de su propia suma directa da error ≈ 0; con un leapfrog sin el segundo medio kick, FALLA; imprime PASA/FALLA por prueba | `S1: verificador` |
| S2 | `impl/py-core` | `py_core/bh_core.py`: Barnes-Hut secuencial. Caja recalculada en cada paso, códigos de Morton, ordenamiento, octree en **arreglos planos** de NumPy con hojas de hasta 8 partículas, recorrido con **pila explícita** que abre siempre la celda que contiene a la partícula, leapfrog, CLI del contrato. Funciones calientes con `@numba.njit(cache=True)`, sin `parallel`/`prange`; el paso de calentamiento no se mide | 1.5 | pasa el verificador; con θ = 1.0 no explota | `S2: Barnes-Hut secuencial en Python` |
| S3 | `impl/mp` | `py_mp/bh_mp.py`: dos modos. `--tree shared`: el principal construye el árbol y lo pone en `multiprocessing.shared_memory`. `--tree local`: cada trabajador reconstruye el árbol. Fuerzas en 4–8 trozos por proceso con `imap_unordered` (balanceo). Numba se compila en el principal antes de crear el `Pool`, con `NUMBA_NUM_THREADS=1`. Escribe `PREFIJO_workers.csv` (trabajador, segundos, interacciones) | 1.5 | ambos modos pasan el verificador; tiempos de los dos con 1, 2, 4 y 8 procesos en `results/mp/` | `S3: multiprocessing, árbol compartido vs local` |
| S4 | `impl/mpi` | `mpi/bh_mpi.py`: caja global con `allreduce`; el rango 0 recibe códigos y partículas, ordena y reparte con `scatterv` (medido como comunicación y ordenamiento: es fracción secuencial); `--particion espacio\|cantidad` (espacio = reparto ingenuo para contrastar en la demo); árbol local; `allgather` de cajas; poda del árbol propio contra cada caja remota usando la **distancia mínima** de la caja al centro de masa; las hojas viajan como partículas; intercambio con `alltoall`; leapfrog; `gather` al rango 0. Escribe `workers.csv` y en `times.csv` los bytes recibidos frente a los del árbol completo | 2.5 | pasa el verificador con `mpirun -n 1, 2 y 4`; los bytes recibidos son menores que el árbol completo | `S4: MPI con Morton y ramas remotas` |
| S5 | `tools/bench` → `main` el vie 27 | Script de bucle (≈20 líneas) que corre una versión con 1, 2, 4, 8 trabajadores y hasta 2× los núcleos lógicos (`mpirun --oversubscribe`), guarda `lscpu` en `results/`, y `plot.py` con tiempo, speedup, eficiencia y fracción secuencial de Karp-Flatt. Una corrida con N = 200 000 por versión | 0.5 | gráficas de tus dos versiones; Sebastián lo puede usar | `S5: mediciones y gráficas` |
| S6 | `docs/informe` | `py_mp/GUIA.md`, `mpi/GUIA.md` y tus secciones del informe (≈1 página por versión, sin pulir) | 1 | cada GUIA cabe en una página | `S6: guías e informe` |
| S7 | — | Estudiar las versiones de Sebastián (GUIA.md y PR) y la hoja de bugs típicos | 2 | respondes sus preguntas de predicción y encuentras los 6 bugs sin mirar | — |

**Decisiones que ya quedan tomadas (para justificar en la sustentación):**

- **Numba** en las funciones calientes: Barnes-Hut en Python puro con 20 000 partículas tarda minutos por paso. Sigue siendo Python; el paralelismo lo dan multiprocessing y MPI, no Numba (sin `prange`, un hilo por proceso). Se compila antes de medir para que la compilación no deforme el speedup. Para depurar sin IA: `NUMBA_DISABLE_JIT=1` y N pequeño. Que Numba cuente como Python es suposición: pregunta al profesor.
- **Reparto por cantidad, no por espacio**: con `corner`, partir el cubo en pedazos iguales le da casi todo a un proceso. Morton + trozos de N/P partículas reparte parejo. El modo `espacio` se deja a propósito para mostrar la diferencia en la demo. Cantidad todavía no es trabajo (un cúmulo abre más celdas): `workers.csv` lo muestra.
- **Ramas podadas en vez del árbol completo**: cada rango recibe de los demás solo las celdas que, vistas desde **el punto más cercano** de su caja, ya cumplen θ. Con la distancia mínima, el resultado es el mismo que el Barnes-Hut secuencial. Los bytes recibidos son la prueba numérica de "sin replicarlo completo". Es Foster: aglomerar para comunicar menos.
- **Pasos para medir**: 20 en vez de 100, reportando tiempo por paso. Suposición a confirmar.

**Si te atrasas:** límite el **viernes 27 en la noche**. Si S4 no pasa el verificador, se activa la simplificación: cada rango envía los resúmenes de sus celdas a una profundidad fija (por ejemplo 3) más las partículas de las celdas vecinas a cada caja remota; sigue sin replicar el árbol. Después se recorta el informe (vale 10%). S7 no se recorta: de ahí sale el 90% de la nota.

## Tareas de Sebastián (C con OpenMP + CUDA)

Son siete tareas que suman ≈11 h, más ≈2 h de encuentros. La base en C (C1) es el corazón: el mismo árbol aplanado sirve para OpenMP y para CUDA. Mientras llega el verificador de Samuel (día 2), se comprueba con θ = 0 contra una suma directa propia en C.

| Tarea | Rama | Qué se construye | Horas | Hecho cuando | Commit |
| --- | --- | --- | --- | --- | --- |
| C0 | — | Clonar, leer `CONTRATO.md`, probar `gcc -fopenmp`, crear el notebook de Colab con GPU T4 | 0.5 | compila un hola mundo OpenMP y el notebook ve la T4 | `C0: entorno` |
| C1 | `impl/c-core` | `c_core/bh.c` secuencial: lectura, caja recalculada en cada paso, códigos de Morton (21 bits por eje), ordenamiento `(código, índice)`, **octree aplanado** con hojas de hasta 8 partículas (arreglos: hijos, masa, centro de masa, centro, mitad del lado), recorrido con **pila explícita** que abre siempre la celda que contiene a la partícula, leapfrog, CLI y salidas del contrato en el orden original. Incluye `suma_directa()` para depurar | 2 | pasa el verificador; con θ = 1.0 no explota | `C1: Barnes-Hut secuencial en C` |
| C2 | `impl/omp` | `c_omp/bh_omp.c` (usa `c_core/`): códigos de Morton con `parallel for`; **construcción recursiva con `#pragma omp task if(n > N/(8·hilos))`**, que corta por cantidad de partículas y no por nivel; nodos reservados con conteo previo y suma de prefijos (o `omp atomic capture` marcado `// SINCRONIZACIÓN:`); fuerzas con `schedule(runtime)` (por defecto `dynamic,64`, se cambia con `OMP_SCHEDULE` en la demo); integración con `parallel for`. Escribe `PREFIJO_workers.csv` (hilo, segundos, interacciones) | 1.5 | pasa el verificador con 1, 2, 4 y 8 hilos; tiempos de `static` vs `dynamic` con `uniform` y `corner`; reparto de la construcción con `corner` | `C2: OpenMP con árbol paralelo y balanceo` |
| C3 | `impl/cuda` | `cuda/bh_cuda.cu` en Colab (`%%writefile` + `nvcc`): kernel de códigos Morton, `thrust::sort_by_key`, árbol aplanado construido en la CPU con `c_core/` (ver decisión), kernel de recorrido **sin recursión** (pila en memoria local, un hilo por partícula), `ORDEN_HILOS = morton \| original` arriba del archivo, kernel de integración. `cudaEvent` por fase, con la copia CPU→GPU, la construcción y la copia de vuelta dentro de "construcción". `compute-sanitizer --tool memcheck` | 3 | pasa el verificador (umbral float32); `times.csv` con las 4 fases; notebook ejecutado y sin errores de memcheck | `C3: CUDA con árbol aplanado` |
| C4 | `tools/bench` (desde `main`, vie 27) | Con el script de S5: OpenMP con 1, 2, 4, 8 hilos y hasta 2× los núcleos lógicos; CUDA con bloques de 64, 128 y 256 hilos, desglose por fase y `morton` vs `original`. Una corrida con N = 200 000 por versión | 1 | gráficas en `results/omp/` y `results/cuda/` | `C4: mediciones` |
| C5 | `docs/informe` | `c_omp/GUIA.md`, `cuda/GUIA.md` y tus secciones del informe (≈1 página por versión, sin pulir) | 1 | cada GUIA cabe en una página | `C5: guías e informe` |
| C6 | — | Estudiar las versiones de Samuel (GUIA.md y PR) y la hoja de bugs típicos | 2 | respondes sus preguntas de predicción y encuentras los 6 bugs sin mirar | — |

**Decisiones que ya quedan tomadas (para justificar en la sustentación):**

- **Construcción paralela sin candados, cortada por cantidad**: con Morton, cada subárbol ocupa un rango contiguo de partículas y ningún hilo escribe en el de otro. El plan anterior (64 subárboles fijos, nivel 2) fallaba con `corner`: la auditoría midió que un solo subárbol se quedaba con el 95% de las partículas, porque la caja de `corner` cabe en una sola celda de nivel 2. Por eso las tareas se crean según cuántas partículas tiene el nodo, no según su profundidad.
- **`schedule(dynamic, 64)` en las fuerzas**: las partículas de un cúmulo denso abren muchas más celdas que las aisladas. Con `static` los hilos con cúmulos tardan más y los demás esperan. La medición con `corner` y `workers.csv` lo demuestran. `schedule(runtime)` deja cambiarlo en la demo sin recompilar.
- **CUDA, árbol en la CPU**: el enunciado pide árbol aplanado, partículas ordenadas y recorrido sin recursión, pero no dice dónde se construye. Aun así, se pregunta al profesor. Plan B si no lo acepta: árbol radix binario de Karras en la GPU (2–3 h extra; suposición: podría no valer si exigen octree).
- **Divergencia de warp**: con `ORDEN_HILOS = morton`, los 32 hilos de un warp son vecinos en el espacio, abren casi las mismas celdas y recorren el árbol casi igual. Con `original` se ve la diferencia en el desglose.
- **float32 en CUDA**: la T4 es mucho más rápida en FP32 que en FP64. Por eso el umbral del verificador es más laxo para CUDA.
- **`racecheck` no prueba mucho aquí**: solo revisa memoria compartida, y este diseño usa pilas locales. La prueba que importa es `memcheck`. Si lo preguntan en la sustentación, se dice así.

**Si te atrasas:** límite el **viernes 27 en la noche**. Si C3 no pasa el verificador, se simplifica: sin optimización de warps (solo orden Morton) y desglose de tiempos básico. C3 tiene prioridad sobre C2: es la difícil y depende de Colab. Después se recorta el informe (vale 10%). C6 no se recorta.

## Encuentros y estudio cruzado

Solo hay dos encuentros. El estudio de la parte del otro pasa por las GUIA.md, porque en la sustentación y en la demo cualquiera puede recibir preguntas de cualquier versión.

| Día | Samuel | Sebastián | Hito común |
| --- | --- | --- | --- |
| Mié 25 | S0, S1, S2 | C0, C1 | repo creado y verificador subido |
| Jue 26 | S3 | C1 (cierre), C2 | **Encuentro 1 (20 min, virtual):** cada uno muestra una versión pasando el verificador; se revisa `CONTRATO_DUDAS.md` |
| Vie 27 | S4 | C3 | — |
| Sáb 28 | S4 (cierre), S5 | C3 (cierre), C4 | las 4 versiones pasan el verificador |
| Dom 29 | S6, S7 | C5, C6 | **Encuentro 2 (1.5–2 h):** ensayo de sustentación y demo |
| Lun 30 | holgura | holgura | informe unido y entrega |

**Formato de cada GUIA.md (una página, la escribe el dueño de la versión con ayuda de su agente):**

1. Qué se paraleliza, por qué se puede y qué queda secuencial.
2. Dónde podría haber una carrera o un deadlock y cómo se evitó, con archivo y línea.
3. Dónde se cambia cada perilla de la demo (θ, particionado, `schedule`, tamaño de bloque, asignación a warps), con archivo y línea.
4. Tres preguntas de predicción con su respuesta, por ejemplo "qué pasa de 4 a 8 procesos y por qué".
5. Qué se ve en el monitor de recursos con `uniform` y con `corner`, y por qué.

**Encuentro 2, en este orden y sin IA:**

1. Cada uno explica sus dos versiones al otro, 20 min cada uno, sin mirar el código.
2. Simulacro de la parte 2 de la sustentación con la **hoja de 6 bugs típicos**, que no aparecen en el código propio (MPI usa colectivas y CUDA no usa memoria compartida, así que nunca los habrán visto). Cada uno se encuentra a papel con su arreglo:
   1. `Send`/`Send` bloqueante entre dos procesos con mensajes grandes.
   2. Colectivas llamadas en distinto orden en distintos rangos.
   3. `Recv` esperando una etiqueta o un origen equivocado.
   4. Reducción en memoria compartida sin `__syncthreads()`.
   5. `__syncthreads()` dentro de un `if` divergente.
   6. `+=` sobre memoria global sin `atomicAdd`.
3. Simulacro de la demo: correr las 4 versiones con `clusters` (con velocidades y masas aleatorias) y con `corner`, con dt = 2.5e-4 y el monitor abierto, y explicar lo que se ve, apoyados en `workers.csv`.
4. Una modificación en vivo por versión, hecha por quien **no** la programó: cambiar θ (probar θ = 1.0), `--particion espacio` en MPI, `OMP_SCHEDULE` en OpenMP, `ORDEN_HILOS` o el tamaño de bloque en CUDA.

La demo es en **un** computador del grupo. Hay que decidir cuál y probar ahí las 4 versiones antes del Encuentro 2 (MPI necesita WSL2 si es Windows; CUDA corre en Colab desde cualquiera).

**Informe (máximo 6 páginas, PDF):** ≈1 página por versión (4 páginas), 1.5 para las mediciones comunes (escalado, más trabajadores que núcleos con `lscpu`, Karp-Flatt, desglose de la GPU) y 0.5 para la verificación. Las gráficas se comparten entre versiones para ahorrar espacio.

## Instrucciones para los agentes (CLAUDE.md)

Este texto va en la raíz del repositorio como `CLAUDE.md`. Claude Code lo lee solo al abrir el proyecto, así los dos agentes siguen las mismas reglas sin que nadie las repita. Después, cada uno le pide una tarea a la vez: "haz la tarea S3" o "haz la tarea C2".

```markdown
# Proyecto Barnes-Hut — reglas para el agente

## Contexto
Proyecto final de Programación Concurrente y Distribuida. Simulación de N cuerpos
con Barnes-Hut en 4 versiones paralelas. Los humanos deben poder LEER, EXPLICAR y
MODIFICAR este código en pocos minutos, sin IA ni internet, en una demo en vivo.
Eso manda sobre la velocidad y la elegancia.

## Tu rol
- Eres el trabajador: construyes, pruebas y documentas la tarea pedida.
- No enseñas ni haces preguntas de verificación. El estudio se hace en otro
  chat, a partir de lo que dejes escrito en NOTAS.md.

## Fuente de verdad
- CONTRATO.md: formato de archivos, física, CLI y salidas. No se cambia.
- DISEÑO.md: tareas S0–S7 (Samuel) y C0–C6 (Sebastián). Haz SOLO la tarea pedida.

## Dueños de carpetas
- Samuel: common/, py_core/, py_mp/, mpi/, results/<impl>/, informe/samuel_*.md
- Sebastián: c_core/, c_omp/, cuda/, results/<impl>/, informe/sebastian_*.md
- Trabaja SOLO en la rama de la implementación de la tarea (impl/py-core,
  impl/mp, impl/mpi, impl/c-core, impl/omp, impl/cuda). Si no existe, créala
  desde main. Al terminar la versión, abre un PR a main sin squash con la
  salida del verificador en la descripción.
- Nunca edites carpetas del otro. Si el contrato no alcanza, escribe en
  CONTRATO_DUDAS.md y sigue con una suposición marcada en el código.

## Estilo obligatorio
- Comentarios en español que expliquen el PORQUÉ, no el qué.
- Todas las perillas de la demo (theta, eps, dt, particionado, schedule,
  tamaño de bloque) arriba del archivo o en la CLI, con un comentario.
- Funciones cortas (< 40 líneas), nombres en español claros, nada de trucos.
- Marca con `# PARALELO:` / `// PARALELO:` cada punto donde se reparte trabajo
  y con `# SINCRONIZACIÓN:` / `// SINCRONIZACIÓN:` cada punto donde hay
  comunicación, candado, barrera o posible carrera.
- Mide cada fase y escribe PREFIJO_times.csv como dice el contrato.

## Al terminar una tarea
1. Corre `python common/verify.py correr --cmd "<programa>"`. Si algo FALLA,
   la tarea no está hecha.
2. Haz UN commit con el código de la tarea al inicio del mensaje.
3. Agrega al final de `NOTAS.md`, dentro de la carpeta de la implementación
   (en `common/NOTAS.md` para S0, S1 y S5), una entrada con: tarea y fecha,
   archivos creados o cambiados, decisiones y por qué, qué tendría que poder
   explicar el humano en la sustentación (conceptos y líneas clave con
   archivo:línea, posibles carreras o deadlocks), y dudas abiertas.
4. En el chat responde solo con 5 líneas de resumen, la tabla PASA/FALLA y lo
   pendiente. Sin preguntas de verificación.
```

Claude Code es el trabajador y el estudio se hace en otro chat. El puente son los NOTAS.md.

## Riesgos y suposiciones para el auditor

El mayor riesgo sigue siendo el tiempo: el plan pide ≈13 h por persona contando encuentros, frente a 8–10 disponibles. La auditoría del 24-nov (15 hallazgos) ya está aplicada en las secciones de arriba; esto es lo que queda abierto.

| # | Riesgo o suposición | Impacto si falla | Qué hacer |
| --- | --- | --- | --- |
| 1 | CUDA construye el árbol en la CPU | la versión podría no cumplir | preguntar al profesor; plan B: árbol radix de Karras en la GPU (2–3 h) |
| 2 | Las horas de S4 (2.5 h) y C3 (3 h) alcanzan con IA | una versión difícil sin terminar | límite vie 27 en la noche y simplificación ya definida en cada sección de tareas |
| 3 | Umbrales del verificador y dt = 2.5e-4 para distribuciones concentradas | falsos FALLA o falsos PASA | ajustarlos el mié 25 con S2 y dejarlos fijos |
| 4 | Numba se acepta como Python | el profesor podría objetarlo | preguntar; justificar que el paralelismo lo dan multiprocessing y MPI |
| 5 | 20 pasos en vez de 100 para medir | mediciones distintas a las pedidas | reportar tiempo por paso; preguntar |
| 6 | La demo es sin internet y CUDA corre en Colab | no se puede mostrar la GPU | el enunciado prevé explicar con el desglose de tiempos entregado; tenerlo en PDF local; preguntar |
| 7 | La distribución de la demo trae N = 200 000, velocidades o masas distintas | algo que nunca se probó falla en vivo | `gen.py` ya genera esos casos; correr las 4 versiones con N = 200 000 antes del sábado |
| 8 | El computador de la demo corre MPI (WSL2, `--oversubscribe`) y la versión de C | la demo falla por entorno | probarlo el sáb 28 |
| 9 | Fecha de entrega real (no confirmada) | si es antes del 30, no hay holgura | confirmar |

**Para usar este documento en el repo:** exportarlo a Markdown como `DISEÑO.md` en la raíz (el `CLAUDE.md` lo referencia), y copiar la sección "Contrato común" en `CONTRATO.md`.

**Preguntas para el profesor, antes de empezar:**

1. En CUDA, ¿se acepta construir el octree en la CPU, con el ordenamiento Morton, el recorrido y la integración en la GPU, midiendo la construcción y las copias con `cudaEvent`? ¿Y la demo, que es sin internet, permite conectarse a Colab?
2. Sobre la distribución de la demo: ¿qué N tendrá, cuántos pasos se corren, y trae velocidades o masas distintas? ¿Se exige que la energía se conserve sobre ella, y podemos ajustar dt y ε?
3. ¿Cuenta como "Python con multiprocessing" usar Numba (sin paralelismo propio) en las funciones calientes? ¿Se puede medir con 20 pasos y reportar el tiempo por paso?
4. ¿Cuál es la fecha exacta de entrega y de sustentación?
