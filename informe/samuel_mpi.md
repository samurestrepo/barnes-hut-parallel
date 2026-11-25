# MPI con mpi4py

**Diseño.** `mpi/bh_mpi.py` reutiliza el árbol, el recorrido y el leapfrog de `py_core/`. En cada
paso, 11 colectivas en el mismo orden en todos los rangos (así no puede haber deadlock por orden) y
ningún envío punto a punto:

1. `Allreduce` de mínimos y máximos: todos usan la misma caja para los códigos de Morton.
2. El rango 0 junta códigos y partículas (`Gatherv`), las ordena, corta en P trozos y reparte
   (`Scatterv`). Este ordenamiento es la fracción secuencial de la versión.
3. Cada rango construye el árbol de su trozo y, con `Allgather`, todos conocen la caja de todos.
4. **Poda**: cada rango recorre su árbol contra la caja de cada rango remoto. Una celda que cumple θ
   vista desde el punto **más cercano** de esa caja lo cumple para cualquier partícula de ella y
   viaja como (centro de masa, masa); si no, se abre, y las hojas abiertas viajan como partículas.
   Intercambio con `Alltoall` de cuentas y `Alltoallv`.
5. Cada rango arma un árbol con lo suyo más lo recibido, calcula las aceleraciones de sus
   partículas e integra; al final, `Gatherv` al rango 0 para escribir.

**Sin replicar el árbol** (lo que pide el enunciado). Cada rango recibe solo lo que la poda deja
pasar: 27–33% de lo que llegaría con el árbol completo con N = 20 000 y 15–18% con 200 000, con 2,
4 y 8 rangos (`bytes_*` de `times.csv`, iguales en Windows y en el computador de la demo). Con más
partículas la proporción baja: cada celda lejana resume más partículas.

**Particionado.** `cantidad` (N/P partículas por rango en orden Morton) es el modo por defecto. Con
`espacio` (tramos de igual volumen) el rincón denso de `corner` cae entero en un rango y los demás
esperan (ver Mediciones comunes). Pero `cantidad` reparte partículas, no trabajo: cuando el rincón
colapsa, los rangos se desbalancean, mientras multiprocessing, con reparto dinámico, sigue parejo.
Las tablas de Windows están en `mpi/NOTAS.md`.

**Hallazgo con Microsoft MPI (Windows).** Con más rangos que núcleos, 8 rangos tardaban 9.6 s,
8 de ellos en "comunicación". Hipótesis: los rangos que esperan en una
colectiva hacen espera activa y le quitan CPU a los que calculan. Lo medido: `MPICH_PROGRESS_SPIN_LIMIT=16` (arriba del
archivo) lo baja a 2.27 s. Con Open MPI en el computador de la demo no aparece: con 32 rangos en
16 hilos la comunicación crece con la sobresuscripción (0.71 s con N = 20 000), sin ese salto.

*(Escalado en el computador de la demo: ver Mediciones comunes.)*
