# Notas de `c_core/` (C1)

Bitácora de lo que se construyó, para estudiarlo. Las entradas nuevas van al final.

---

## C1 — Barnes-Hut secuencial en C (2026-11-24)

**Archivos creados:**
- `c_core/bh_core.h`: estructuras (`Particulas`, `Caja`, `Arbol`, `Tiempos`, `Parametros`), perillas fijas (`HOJA_MAX`, `BITS_EJE`, `PILA_MAX`) y prototipos. Es lo que incluirán `c_omp/` y `cuda/`.
- `c_core/bh_arbol.c`: caja, códigos de Morton, ordenamiento y construcción del octree aplanado.
- `c_core/bh_fuerzas.c`: recorrido con pila explícita, `suma_directa()` y leapfrog.
- `c_core/bh_io.c`: CLI del contrato, lectura, escritura con `%.17g`, reloj y memoria.
- `c_core/bh.c`: `main` secuencial (el bucle de pasos y la medición por fase).
- `c_core/Makefile`: `make -C c_core` deja el ejecutable `c_core/bh`.

**Resultado del verificador** (`python common/verify.py correr --cmd "./c_core/bh"`):

| Prueba | Resultado | Valor | Umbral |
| --- | --- | --- | --- |
| fuerzas_directa | PASA | 1.4e-14 | 1e-6 |
| fuerzas_bh | PASA | 1.7e-3 | 1e-2 |
| trayectoria | PASA | 2.8e-17 | 1e-6 |
| energia_reposo | PASA | 3.4e-8 | 1e-5 |
| energia_movimiento | FALLA | 6.1e-3 | 1e-3 |

`energia_movimiento` falla por el umbral, no por un error de código (ver `CONTRATO_DUDAS.md`, 24-nov, C1):
- con θ = 0 la deriva es 3.982e-4, **idéntica** a la de `directo.py`;
- con `--directa` también da 3.982e-4;
- las fuerzas con θ = 0.5 en esa distribución tienen mediana 2.4e-3 y máximo 1.2% (correctas para θ = 0.5);
- dividir dt a la mitad no cambia la deriva (6.4e-3), así que no es el integrador: es el error de aproximar celdas;
- con θ = 0.3 baja a 6.8e-4.

Otras pruebas: θ = 1.0 con `uniform` 100 pasos da deriva 1.3e-6 (no explota). `corner` con N = 20 000, posiciones fuera de [0, 1] (`--escala 5`) y N = 200 000 corren sin errores. Con `-fsanitize=address,undefined` no hay errores de memoria. N = 200 000: ≈ 2.9 s por paso (5.5 s de recorrido en 2 pasos + el paso 0), y el recorrido es el 90% del tiempo.

**Decisiones y por qué:**
- **Partículas en arreglos separados (SoA)** (`bh_core.h`, struct `Particulas`): `x[i]`, `y[i]`... en vez de un struct por partícula. Los hilos vecinos de OpenMP y de CUDA leen memoria contigua.
- **Morton con un bucle** (`bh_arbol.c:45`, `separar_bits`): pone el bit b en la posición 3b. Hay versiones con "números mágicos" más rápidas, pero esta se explica en una línea y el ordenamiento no es el cuello de botella.
- **Ordenar y mover los datos** (`bh_arbol.c:92`): después de ordenar, las partículas de cada celda quedan contiguas. Cada nodo guarda solo `inicio` y `cuenta`; las hojas se suman con un `for` sobre ese rango. `id[]` se mueve con ellas para escribir en el orden original (`bh_io.c:112`).
- **Construir partiendo rangos, no insertando** (`bh_arbol.c:158`): como el arreglo está ordenado, el octante de nivel `nivel` es un tramo contiguo; se avanza mientras `octante(codigo, nivel)` no cambie (`bh_arbol.c:68`). La construcción es recursiva (máximo 22 niveles); en C2 cada hijo será una `task`.
- **Hojas de hasta 8, salvo en el nivel 21** (`bh_arbol.c:180`): en el nivel 21 ya no quedan bits, todas las partículas comparten código y quedan en una hoja aunque sean más de 8.
- **Recorrido con pila explícita** (`bh_fuerzas.c:37`): igual que el kernel de CUDA, que no puede usar bien la recursión. `PILA_MAX = 256` alcanza: 22 niveles × 7 hermanos pendientes como máximo.
- **Abrir siempre la celda que contiene a la partícula** (`bh_fuerzas.c:21` y `:49`): si no, con θ grande la partícula se atraería a sí misma a través del centro de masa de su celda.
- **El árbol reutiliza su memoria** (`bh_arbol.c:110`, `crecer_arbol`): se duplica solo cuando falta; en el paso siguiente `n_nodos` vuelve a 0.
- **Fases medidas** (`bh.c:15`): `ordenamiento` = caja + códigos + `qsort` + mover datos; `construccion` = árbol con masas; `recorrido` = aceleraciones; `integracion` = kicks y drift; `comunicacion` = 0 (un solo proceso).

**Qué tendría que poder explicar el humano en la sustentación:**
- Por qué Morton hace que cada celda sea un rango contiguo, y cómo se lee el octante de un nivel: `(codigo >> 3*(20 - nivel)) & 7`, con el bit x en la posición 2 (`bh_arbol.c:68`). El centro del hijo se calcula con esos bits (`o & 4` → x, `o & 2` → y, `o & 1` → z).
- El criterio de apertura `lado < θ · distancia` (`bh_fuerzas.c:49`) y que θ = 0 lo vuelve suma directa (por eso `fuerzas_directa` da 1.4e-14 y no 0: solo cambia el orden de las sumas).
- El leapfrog KDK (`bh.c:32`): medio kick con la aceleración vieja, drift, reconstrucción del árbol, medio kick con la nueva.
- `a->hijos[8*k + o] = hijo` va **después** de la llamada recursiva (`bh_arbol.c:170`): la llamada puede hacer `realloc` y mover `a->hijos`. En C2 eso sería una carrera si dos hilos crearan nodos a la vez: por eso allí los nodos se reservan antes o con `omp atomic capture`.
- Dónde se paraleliza después (marcas `PARALELO`): códigos (`bh_arbol.c:60`), construcción de hijos (`bh_arbol.c:168`) y aceleraciones (`bh_fuerzas.c:60`). En las aceleraciones no hay carrera: cada `i` lee el árbol y escribe solo `ax[i]`, `ay[i]`, `az[i]`.
- Aquí no hay concurrencia ni deadlocks: es la versión secuencial.

**Dudas abiertas:**
- Umbral de `energia_movimiento` con θ = 0.5 (anotado en `CONTRATO_DUDAS.md`). S2 debería ver lo mismo en Python.
