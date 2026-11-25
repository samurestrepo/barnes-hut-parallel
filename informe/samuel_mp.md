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

**Compartir o reconstruir el árbol** (lo que pide el enunciado, decidido con mediciones):

- `--tree shared`: el principal construye el árbol y lo copia a memoria compartida; los
  trabajadores lo leen sin copiarlo. Cuesta ≈ 0.3 ms por paso.
- `--tree local`: cada trabajador reconstruye el mismo árbol en cada paso. Con N = 200 000 la
  construcción pasa de 1.4 s con 1 proceso a 4.7 s con 8, porque se repite P veces y los procesos
  compiten por la memoria.

`shared` gana o empata en todos los casos medidos y es el modo por defecto
(`results/mp/uniform_N*_bench.csv`).

**Hallazgos.**

- **El mayor sobrecosto es crear el `Pool`**: de 1.8 a 7.2 s según el número de procesos, porque
  cada proceso arranca de cero (*spawn* en Windows) e importa NumPy y Numba. Por eso se crea una
  vez y no en cada paso. Por la cola solo viajan tuplas pequeñas (índices de trozo).
- **Balanceo**: con `corner` y 4 procesos los trabajadores terminan dentro del 1–2% en tiempo e
  interacciones (`results/mp/corner_N20000_shared_w4_workers.csv`), y sigue parejo a 100 pasos,
  cuando el rincón ya colapsó (11.11–11.17 s, `results/demo/`).

*(Escalado en el computador de la demo: ver Mediciones comunes.)*
