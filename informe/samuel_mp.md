# Python con multiprocessing

**Diseño.** El núcleo `py_core/bh_core.py` arma el octree en arreglos planos de NumPy y compila con
Numba las funciones calientes, sin paralelismo propio: un hilo por proceso, el paralelismo lo da
`multiprocessing`. Numba se compila en el proceso principal antes de crear el `Pool` y el paso de
calentamiento no se mide. `py_mp/bh_mp.py` reparte el cálculo de fuerzas, que en la versión
secuencial es el 95% del tiempo. Posiciones, masas y aceleraciones viven en
`multiprocessing.shared_memory`; las fuerzas se parten en 6 trozos por proceso, contiguos en orden
Morton, que `imap_unordered` entrega al primer proceso libre (reparto dinámico). No hay candados:
cada trozo escribe filas distintas de la aceleración, y terminar de consumir `imap_unordered` hace
de barrera antes de integrar. El `Pool` se crea una sola vez y la memoria compartida se libera en
un `finally`.

**Compartir o reconstruir el árbol** (lo que pide el enunciado). Con `--tree shared` el principal
construye el árbol y lo copia a memoria compartida (≈ 0.3 ms por paso); los trabajadores lo leen
sin copiarlo. Con `--tree local` cada trabajador reconstruye el mismo árbol en cada paso: la
construcción se repite P veces y los procesos compiten por la memoria (ver Mediciones comunes).
`shared` gana o empata en todos los casos medidos y es el modo por defecto.

**Hallazgos.** Crear el `Pool` es el mayor sobrecosto (cada proceso importa NumPy y Numba), por eso
se crea una sola vez. Repartir los trozos también cuesta en cada paso: la fase `recorrido` incluye
ese reparto, y con θ = 1.0 en 8 trabajadores el cálculo baja de 0.36 a 0.09 s mientras `recorrido`
se queda en ≈ 0.42 s. Con `corner` el reparto dinámico deja a los trabajadores dentro del 1–2%,
también a 100 pasos, cuando el rincón ya colapsó (`results/mp/`, `results/demo_pc/ensayo/`).
