# Mediciones comunes

**Método.** CPU en el computador de la demo (i7-11800H, 8 núcleos físicos / 16 hilos, WSL2,
Open MPI 5.0.10; `results/demo_pc/`, `results/omp/`); CUDA en Colab (T4, desglose en su sección).
`uniform`, θ = 0.5, 20 pasos; 3 repeticiones con N = 20 000 y una con 200 000.Todo desde el disco de
WSL (ver `/mnt/c` en OpenMP).

**Cada versión contra el T1 de su lenguaje.** S2 tarda 1.92 / 28.3 s (N = 20 000 / 200 000) y C1
3.69 / 45.5 s. Python no es más rápido: hace menos trabajo (197 frente a 440 millones de
interacciones con N = 20 000; 2 982 frente a 6 115 con 200 000) por la diferencia en las hojas
explicada en Verificación. Por interacción, C es 1.2–1.3 veces más rápido.

![](results/demo_pc/conjunta_N20000_escalado.png){width=72%}\

*Figura 1. Escalado con N = 20 000 (N = 200 000: `results/demo_pc/conjunta_N200000_escalado.png`).*

**Hasta 8.** Speedup con 8 y N = 200 000: OpenMP 4.50, mp shared 4.25, mp local 3.99, MPI 3.58
(N = 20 000: 5.42, 4.12, 3.52, 3.13).

**Más procesos o hilos que núcleos.** Con N = 20 000 OpenMP sube de 8 a 16 (hyperthreading) y las
de Python bajan: hay ≈ 0.5 s de cálculo y el reparto pesa. Con 200 000, mp shared y MPI mejoran
hasta 16 (5.55 y 4.51). Con 32 ninguna mejora: más procesos que hilos de hardware solo agregan
cambios de contexto. Crear el `Pool` cuesta 0.43 s con 1 proceso y 1.2 s con 32 (*fork*; en
Windows, con *spawn*, 1.8–7.2 s), fuera del total.

**Por qué se aplanan las curvas.**

- **Fracción secuencial (Amdahl):** en MPI, Karp-Flatt ≈ 0.17 casi constante de 2 a 16 rangos
  (N = 200 000): el rango 0 junta, ordena y reparte en cada paso.
- **Sobrecosto que crece con p:** mp local reconstruye el árbol en cada proceso (0.75 s con 1,
  4.40 s con 32); por eso shared ≥ local. La comunicación de MPI pasa de 0.05 s (2) a 0.71 s (32).
- **Memoria:** con N = 200 000 el árbol no cabe en caché y OpenMP escala peor que con 20 000.
- **Desbalance** (ensayo con `corner`, 100 pasos, 8 trabajadores): mp con `imap_unordered` deja a
  los 8 dentro del 1% (speedup 4.29); MPI `cantidad`, de 1.45 a 3.13 s por rango (2.72); MPI
  `espacio` deja ≈ 19 120 de 20 000 partículas en el rango 0 y tarda 20.3 s, más que S2 (13.2 s).
