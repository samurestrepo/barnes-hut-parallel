# Dudas y problemas con el contrato

Si algo de `CONTRATO.md` no funciona, se anota aquí y se sigue trabajando con una suposición marcada en el código. Se revisa en el Encuentro 1 (jue 26).

| Fecha | Quién | Tarea | Problema | Suposición mientras tanto | Resuelto |
| --- | --- | --- | --- | --- | --- |
| 2026-11-24 | Samuel | S1 | El contrato no fija cuántas cifras llevan las salidas. Con `%.6e` el redondeo del texto (~5e-7) ya roza los umbrales de 1e-6 en fuerzas y trayectoria. | Escribir `acc0.txt` y `final.txt` con 17 cifras significativas: `%.17g` en C y `fmt="%.17g"` en `np.savetxt`. CUDA en float32 puede usar `%.9g`. | sí, 24-nov: fijado `%.17g` en CONTRATO.md (Salidas obligatorias) |
| 2026-11-24 | Samuel | S1 | `energia_movimiento` (clusters con velocidad 0.3 y masas aleatorias): la suma directa ya da una deriva de 4.0e-4 con el umbral en 1e-3. Barnes-Hut con θ = 0.5 suma su propio error. | Se deja 1e-3 y se ajusta con S2, como dice el riesgo 3 del diseño. | no |
