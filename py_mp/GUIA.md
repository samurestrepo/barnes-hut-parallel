# GUIA — Barnes-Hut con multiprocessing (`py_mp/bh_mp.py`)

Una página para estudiar la versión sin reunirse. Detalle y números en `py_mp/NOTAS.md`.

```
python py_mp/bh_mp.py --tree shared --workers 4 --input data/uniform_N20000_s42.txt --steps 20 --out salida/mp
```

## 1. Qué se paraleliza, por qué se puede y qué queda secuencial

- **Se paralelizan las fuerzas** (95% del tiempo en S2). La aceleración de cada partícula solo **lee** el árbol y las posiciones, y cada una se escribe en su propia fila: las partículas son independientes. Se parten en 6 × workers trozos contiguos en orden Morton y se reparten con `imap_unordered` (`py_mp/bh_mp.py:137`): el primer proceso libre toma el siguiente trozo (balanceo dinámico).
- **`--tree shared`**: el principal construye el árbol (secuencial) y lo copia a `multiprocessing.shared_memory` (`publicar_arbol`, `:98`); los trabajadores lo leen sin copiarlo.
- **`--tree local`**: cada trabajador reconstruye el mismo árbol una vez por paso (`arbol_local`, `:185`): no hay copia, pero el trabajo se repite p veces.
- **Queda secuencial**: en shared, ordenar y construir el árbol en el principal (≈ 8% con 4 procesos); en los dos modos, la integración (0.3%) y la lectura. Crear el Pool (1.8–7.2 s en Windows) se hace **una vez** (`:276`).

## 2. Carreras y deadlocks: dónde podrían estar y cómo se evitaron

| Riesgo | Dónde | Por qué no pasa |
| --- | --- | --- |
| Dos trozos escriben la misma fila de `acc` | `py_mp/bh_mp.py:222` | `orden` es una permutación y los rangos [ini, fin) no se tocan: filas disjuntas, sin candado |
| El principal mueve `pos` mientras un trabajador la lee | `:140`, `:189` | el `for` de `imap_unordered` es la barrera: se integra solo cuando llegaron TODOS los trozos |
| Liberar la memoria compartida con trabajadores vivos | `:285` | `close` + `unlink` en un `finally`, después de cerrar el `with Pool` |
| Un trabajador borra un bloque del principal al salir | `:59` | se abre con `track=False`: el dueño es el principal |
| Deadlock | — | no hay candados; la única espera es el principal esperando tareas que siempre terminan. Si una falla, `imap_unordered` relanza la excepción y el `finally` libera todo |

## 3. Perillas de la demo

| Perilla | Dónde | Qué cambia |
| --- | --- | --- |
| θ | `--theta` (`py_mp/bh_mp.py:235`); por defecto `py_core/bh_core.py:29` | θ = 1.0: menos interacciones, más rápido, menos preciso; θ = 0: suma directa |
| árbol compartido / local | `--tree shared\|local` (`:238`) | local repite la construcción en cada proceso |
| procesos | `--workers K` (`:237`) | más de 4 = sobresuscripción en esta máquina (2 núcleos, 4 hilos) |
| trozos por proceso | `TROZOS_POR_PROCESO = 6` (`:40`) | 1 = un trozo por proceso (reparto estático); con `corner` igual queda parejo (1.82–1.84 s) |
| dt, ε | `--dt`, `--eps` (`:234`, `:236`) | con `corner`/`clusters` usar dt = 2.5e-4 |

## 4. Tres preguntas de predicción

1. **¿Qué pasa de 4 a 8 procesos?** Empeora un poco (shared 2.04 → 2.18 s): la máquina tiene 4 hilos de hardware, así que 8 procesos se turnan, y crear el Pool sube de 3.8 a 7.2 s.
2. **¿Con 1 proceso, `--tree shared` es tan rápido como S2?** No: 16% más lento (3.84 vs 3.32 s), por repartir 6 trozos por paso por la cola del Pool y escribir `acc` con índices. Es el costo fijo de la maquinaria.
3. **Con `corner` y 4 procesos, ¿un trabajador tarda mucho más que los otros?** No: quedan dentro del 2% (1.87–1.90 s). Y, contra lo que se esperaría, **también con `TROZOS_POR_PROCESO = 1`** (1.82–1.84 s, `results/mp/corner_N20000_shared_w4_trozos1_workers.csv`): con 4 trozos de N/4 en orden Morton, los cuatro caen casi enteros en el rincón denso y hacen el mismo trabajo (60–63 millones de interacciones). El balanceo dinámico protege de casos peores, pero con `corner` no hace falta.

## 5. Qué se ve en el monitor de recursos

Medido con `wmic` cada ≈ 0.6 s, N = 20 000, 100 pasos, `--tree shared --workers 4` (`results/mp/monitor_cpu.csv`; 100% = un hilo de hardware, máximo 400%):

- **Los primeros ≈ 5 s**: un proceso al 60–100% (el principal leyendo y compilando) y luego 4 procesos al 45–70% que arrancan (el Pool: cada uno importa NumPy y carga Numba).
- **Con `uniform`**: 4 procesos al 60–95% más el principal al 10–30% (construye y copia el árbol cada paso); total ≈ 270–340%. No llega a 400% porque son 2 núcleos con hyper-threading y el principal construye mientras los trabajadores esperan.
- **Con `corner`** (dt = 2.5e-4): igual de parejo, 4 procesos al 70–90% (total 290–360%). El rincón denso no se nota porque los trozos son pequeños y dinámicos; `workers.csv` lo confirma (1.87–1.90 s por trabajador).
- Al final, un solo proceso (el principal escribiendo las salidas).
