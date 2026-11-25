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

**Sin replicar el árbol** (lo que pide el enunciado). Bytes recibidos frente a lo que llegaría si
cada rango mandara su árbol completo (`results/mpi/`, filas `bytes_*`):

| Rangos | N = 20 000 | N = 200 000 |
| --- | --- | --- |
| 2 | 32.8% | 15.4% |
| 4 | 30.9% | 17.7% |
| 8 | 27.1% | 15.6% |

Con más partículas la proporción baja: cada celda lejana resume más partículas.

**Particionado: cantidad frente a espacio** (`corner`, N = 20 000, 4 rangos, 20 pasos):

| `--particion` | Total | Partículas por rango | Recorrido por rango |
| --- | --- | --- | --- |
| cantidad (Morton, N/P cada uno) | 2.40 s | 5000 cada uno | 1.60 a 1.79 s |
| espacio (tramos de igual volumen) | 4.75 s | 19 254 / 241 / 252 / 253 | 4.16 / 0.03 / 0.03 / 0.03 s |

Con `espacio`, el rincón denso cae entero en un rango y los otros tres esperan. `cantidad` reparte
partículas, no trabajo: a 100 pasos, cuando el rincón colapsa, los rangos van de 7.1 a 11.4 s
(`results/demo/`), mientras multiprocessing, con reparto dinámico, sigue parejo.

**Hallazgo con Microsoft MPI (Windows).** Con más rangos que núcleos, 8 rangos tardaban 9.6 s,
8 de ellos en "comunicación". Hipótesis: los rangos que esperan en una colectiva hacen espera activa
y le quitan CPU a los que calculan. Lo medido es que `MPICH_PROGRESS_SPIN_LIMIT=16` (arriba del
archivo) lo baja a 2.27 s. Con Open MPI en el computador de la demo no aparece: con 32 rangos en
16 hilos la comunicación crece con la sobresuscripción (0.71 s con N = 20 000), sin ese salto.

*(Escalado en el computador de la demo: ver Mediciones comunes.)*
