# Barnes-Hut con multiprocessing (Python)

*Borrador sin pulir. Todos los números salen de `results/`; la fuente va entre paréntesis.*

## Diseño

El núcleo secuencial (`py_core/bh_core.py`) construye en cada paso la caja de las partículas, sus códigos de Morton de 21 bits por eje, las ordena y arma un octree en arreglos planos de NumPy (hojas de hasta 8 partículas). El recorrido usa una pila explícita y abre siempre la celda que contiene a la partícula. Las funciones calientes van compiladas con Numba sin paralelismo propio (un hilo por proceso): el paralelismo lo da `multiprocessing`.

`py_mp/bh_mp.py` reparte el cálculo de fuerzas, que en la versión secuencial es el 94.8% del tiempo (`results/py_core/resumen_uniform_N20000.csv`). Las posiciones, las masas y las aceleraciones viven en `multiprocessing.shared_memory`; las fuerzas se parten en 6 trozos por proceso, contiguos en orden Morton, que se entregan con `imap_unordered` al primer proceso libre. Hay dos modos:

- **`--tree shared`**: el principal construye el árbol y lo copia a memoria compartida; los trabajadores lo leen sin copiarlo.
- **`--tree local`**: cada trabajador reconstruye el mismo árbol una vez por paso.

No hay candados: cada trozo escribe filas distintas de la aceleración, y el `for` sobre `imap_unordered` hace de barrera antes de integrar.

## Verificación

Con 1 y 4 procesos, en los dos modos, las salidas son idénticas bit a bit a las de la versión secuencial: fuerzas con θ = 0, error relativo máximo 1.5e-14; con θ = 0.5, mediana 3.3e-3; trayectoria de 10 pasos, 2.8e-17; energía en reposo, 3.4e-8. La prueba de energía en movimiento da 8.9e-3 frente al umbral provisional de 1e-3. No es un error de integración: con θ = 0 da 4.0e-4, igual que la suma directa, y a tiempo físico fijo no baja al reducir dt (`results/py_core/deriva_energia.csv`). Se propuso un umbral de 2e-2 (`CONTRATO_DUDAS.md`).

## Rendimiento

Máquina: Intel i5-6300U, 2 núcleos físicos y 4 lógicos (`results/lscpu.txt`, local). Base: la versión secuencial, 3.315 s con N = 20 000 y 52.38 s con N = 200 000 (20 pasos, θ = 0.5). Gráficas: `results/mp/uniform_N20000_escalado.png` y `results/mp/uniform_N200000_escalado.png`.

| N | Modo | S(1) | S(2) | S(4) | S(8) |
| --- | --- | --- | --- | --- | --- |
| 20 000 | shared | 0.90 | 1.31 | 1.84 | 1.73 |
| 20 000 | local | 0.95 | 1.31 | 1.77 | 1.43 |
| 200 000 | shared | 0.97 | 1.55 | 2.11 | 2.11 |
| 200 000 | local | 0.98 | 1.62 | 2.07 | 1.77 |

(`results/mp/uniform_N*_escalado.csv`)

- **El techo es el hardware.** Con 2 núcleos físicos, pasar de 2 a 4 procesos solo agrega hyper-threading; con 8 hay sobresuscripción. Con N = 200 000 el recorrido tarda 22.2 s con 4 procesos y 21.8 s con 8 (`results/mp/uniform_N200000_shared_bench.csv`). La fracción de Karp-Flatt (0.29–0.40 con N = 200 000) es mucho mayor que la parte secuencial medida (≈ 3% de ordenamiento y construcción): mide sobre todo esos núcleos que faltan.
- **shared es mejor o igual que local.** Copiar el árbol cuesta ≈ 0.3 ms por paso (`comunicacion`, 0.006 s en 21 cálculos de fuerza, `results/mp/uniform_N20000_shared_w*_r*_times.csv`). En cambio, local repite la construcción en cada proceso: con N = 200 000 pasa de 1.4 s con 1 proceso a 4.7 s con 8 (`results/mp/uniform_N200000_local_bench.csv`). Con 8 procesos local pierde más (1.77 frente a 2.11).
- **El mayor overhead es crear el Pool**: 1.7 a 5.1 s con N = 200 000 y 1.8 a 7.2 s con N = 20 000, según el número de procesos (fila `crear_pool` de times.csv). En Windows los procesos se crean con *spawn* y cada uno importa NumPy y carga Numba. Por eso el Pool se crea una sola vez. Serializar casi no cuesta: por la cola viajan solo tuplas pequeñas.
- **Balanceo.** Con `corner` (95% de las partículas en un rincón) y 4 procesos, los trabajadores quedan dentro del 2%: 1.87 a 1.90 s y 60.5 a 62.2 millones de interacciones cada uno (`results/mp/corner_N20000_shared_w4_workers.csv`). Con un solo trozo por proceso (reparto estático) queda igual de parejo, 1.82 a 1.84 s (`results/mp/corner_N20000_shared_w4_trozos1_workers.csv`): los cuatro trozos de N/4 en orden Morton caen casi enteros en el rincón y hacen el mismo trabajo. El reparto dinámico es una protección para casos peores, pero con `corner` no hizo falta.
- **Con N grande escala mejor** (2.11 frente a 1.84 con 4 procesos): el trabajo paralelo crece como N log N y los costos fijos pesan menos.

Las mediciones varían entre tandas en este portátil (por ejemplo, 2.04 s y 1.81 s para la misma configuración), así que las versiones se comparan dentro de la misma tanda.
