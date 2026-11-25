# Dudas y problemas con el contrato

Si algo de `CONTRATO.md` no funciona, se anota aquí y se sigue trabajando con una suposición marcada en el código. Se revisa en el Encuentro 1 (jue 26).

| Fecha | Quién | Tarea | Problema | Suposición mientras tanto | Resuelto |
| --- | --- | --- | --- | --- | --- |
| 2026-11-24 | Samuel | S1 | El contrato no fija cuántas cifras llevan las salidas. Con `%.6e` el redondeo del texto (~5e-7) ya roza los umbrales de 1e-6 en fuerzas y trayectoria. | Escribir `acc0.txt` y `final.txt` con 17 cifras significativas: `%.17g` en C y `fmt="%.17g"` en `np.savetxt`. CUDA en float32 puede usar `%.9g`. | sí, 24-nov: fijado `%.17g` en CONTRATO.md (Salidas obligatorias) |
| 2026-11-24 | Samuel | S1 | `energia_movimiento` (clusters con velocidad 0.3 y masas aleatorias): la suma directa ya da una deriva de 4.0e-4 con el umbral en 1e-3. Barnes-Hut con θ = 0.5 suma su propio error. | Se deja 1e-3 y se ajusta con S2, como dice el riesgo 3 del diseño. | no |
| 2026-11-24 | Samuel | S2 | Medido con S2 (`results/py_core/deriva_energia.csv`): `energia_movimiento` da 8.9e-3 con θ = 0.5 y 1.9e-3 con θ = 0.3; con θ = 0 da 4.0e-4, igual que la suma directa. A tiempo físico fijo no baja con dt (7.6e-3 → 8.9e-3 → 9.2e-3 con dt = 5e-4, 2.5e-4, 1.25e-4): es el error de fuerza de Barnes-Hut de monopolo (mediana 3.6e-3) durante el colapso de los cúmulos fríos, no el integrador. Con 1e-3 ninguna versión correcta con θ = 0.5 puede pasar. | **Propuesta: umbral 2e-2** (float64 y float32). Es 2.3× lo medido con θ = 0.5 y sigue atrapando el leapfrog sin segundo medio kick (2.0e-1). Los errores de integrador ya los atrapan `trayectoria` y `energia_reposo`. No se cambió `verify.py`: hasta decidirlo, las versiones marcan FALLA en esa prueba con ≈ 8.9e-3. | no |
