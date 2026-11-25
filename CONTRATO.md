# Contrato común — Barnes-Hut

Copia de la sección "Contrato común" de DISEÑO.md (Nov 24, 2026). No se cambia sin avisar por chat; los problemas van en CONTRATO_DUDAS.md.

Las cuatro versiones leen el mismo archivo, usan la misma física, aceptan los mismos parámetros y escriben las mismas salidas. Con eso, un solo verificador sirve para todas y nadie depende del código del otro.

**Física (fija):**

| Parámetro | Valor | Nota |
| --- | --- | --- |
| G | 1 | del enunciado |
| ε (suavizado) | 0.01 | ajustable por CLI |
| dt | 0.001 con `uniform`; **2.5e-4 con distribuciones concentradas** (`corner`, `clusters`, la de la demo) | con `corner` y dt = 0.001 la energía no se conserva aunque el código esté bien (auditoría: deriva 2.7e-2; con 2.5e-4 baja a 2.0e-4) |
| θ | 0.5 por defecto | lado de la celda / distancia al centro de masa < θ → la celda cuenta como una partícula. **Si la partícula está dentro de la caja de la celda, la celda se abre siempre** (con θ ≳ 0.58 se sumaría a sí misma; la demo puede pedir θ = 1) |
| Hojas | hasta 8 partículas por hoja | Morton de 21 bits puede repetir códigos |
| Caja raíz | min/max de las posiciones, recalculada en cada paso | nunca suponer el cubo unitario: la demo puede traer posiciones fuera de \[0, 1\] |
| Integrador | leapfrog kick-drift-kick | medio paso de velocidad, paso completo de posición, recalcular aceleraciones, medio paso de velocidad |
| Pasos | 10 para verificar; 20 para medir (se reporta tiempo por paso); 100 si alcanza el tiempo | 20 en vez de 100: suposición a confirmar con el profesor |

Aceleración sobre la partícula i (se compara aceleración, no fuerza, para no depender de la masa de i):

```latex
a_i = \sum_{j \neq i} G\, m_j \, \frac{r_j - r_i}{\left(|r_j - r_i|^2 + \varepsilon^2\right)^{3/2}}
```

Energía total (la calcula solo el verificador):

```latex
E = \sum_i \tfrac{1}{2} m_i |v_i|^2 \; - \sum_{i<j} \frac{G\, m_i m_j}{\sqrt{|r_i - r_j|^2 + \varepsilon^2}}
```

**Entrada:** archivo de texto, N filas × 7 columnas separadas por espacio (`x y z vx vy vz m`), float64, sin encabezado. Nombre: `data/<dist>_N<N>_s<semilla>.txt`.

**Parámetros de línea de comandos (iguales en las 4 versiones):**

```
<programa> --input F --steps S --dt 0.001 --theta 0.5 --eps 0.01 --workers K --out PREFIJO
```

`--workers` = procesos (multiprocessing), rangos (MPI: lo da `mpirun -n`, el parámetro se ignora), hilos (OpenMP) o hilos por bloque (CUDA).

**Salidas obligatorias:**

| Archivo | Contenido | Para qué |
| --- | --- | --- |
| `PREFIJO_acc0.txt` | N×3 aceleraciones del paso 0, **en el orden de la entrada** | comparar contra la suma directa |
| `PREFIJO_final.txt` | N×7 estado tras S pasos, **en el orden de la entrada** | energía final |
| `PREFIJO_times.csv` | `fase,segundos` con `lectura, construccion, ordenamiento, recorrido, integracion, comunicacion, total` | informe y gráficas |

**Formato numérico:** `acc0.txt` y `final.txt` se escriben con 17 cifras significativas, `%.17g` (en C `fprintf(f, "%.17g ", x)`; en NumPy `np.savetxt(..., fmt="%.17g")`), separadas por espacio. Con menos cifras el redondeo del texto (~5e-7 con `%.6e`) ya roza los umbrales de 1e-6 del verificador. Fijado el 24-nov (ver `CONTRATO_DUDAS.md`).

Ojo: casi todas las versiones reordenan las partículas por Morton. Hay que guardar el índice original y devolverlas a su orden antes de escribir. Si no, el verificador falla aunque la física esté bien.

**Correcta = pasa `common/verify.py` (criterios iniciales, a ajustar con mediciones):**

| Prueba | Datos | Criterio |
| --- | --- | --- |
| Fuerzas, suma directa | `uniform`, N = 2000, θ = 0 | error relativo máximo < 1e-6 (float64) o < 1e-3 (CUDA en float32) |
| Fuerzas, Barnes-Hut | `uniform`, N = 2000, θ = 0.5 | mediana del error relativo < 1e-2 |
| Trayectoria | `uniform`, N = 2000, θ = 0, 10 pasos | `verify.py` integra los mismos 10 pasos con suma directa y compara `final.txt` **posición por posición**: error máximo < 1e-6 (float64) o < 1e-3 (CUDA). Detecta un integrador mal hecho y un orden de salida equivocado |
| Energía, reposo | `uniform`, N = 2000, θ = 0, 100 pasos | deriva < 1e-5 (float64) o < 1e-3 (CUDA) |
| Energía, en movimiento | `clusters` con velocidades y masas aleatorias, N = 2000, θ = 0.5, dt = 2.5e-4, 100 pasos | deriva < 2e-2 (float64 y float32). Fijado el 24-nov con S2: Barnes-Hut de monopolo con θ = 0.5 da ≈ 8.9e-3 por el error de la aproximación (con θ = 0, 4.0e-4, como la suma directa); el leapfrog sin segundo medio kick da 2.0e-1 y sigue fallando. Ver `CONTRATO_DUDAS.md` |

Por qué se endureció (auditoría): con la nube uniforme en reposo, 100 × 0.001 = 0.1 unidades de tiempo y el colapso tarda unas 0.5, así que las partículas casi no se mueven. Con el umbral anterior de 1e-2 pasaban un leapfrog sin el segundo medio kick (5.6e-3) y hasta Euler explícito (2.2e-4); el correcto da 3.4e-8. Además, la energía no depende del orden de las partículas, por eso ahora se compara la trayectoria. `verify.py` también imprime la deriva de cualquier archivo, incluida la distribución de la demo.

**Generador (cualquiera de los dos puede crearlo sin esperar al otro):**

```python
# common/gen.py  —  python common/gen.py --dist clusters --n 20000 --vel 0.3 --masas aleatorias
import argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument("--dist", default="uniform", choices=["uniform", "clusters", "corner"])
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--vel", type=float, default=0.0)       # 0 = reposo; >0 = velocidades aleatorias
p.add_argument("--masas", default="iguales", choices=["iguales", "aleatorias"])
p.add_argument("--escala", type=float, default=1.0)    # lado del cubo; prueba posiciones fuera de [0, 1]
a = p.parse_args()
rng = np.random.default_rng(a.seed)
if a.dist == "uniform":            # requisito del enunciado
    pos = rng.random((a.n, 3))
elif a.dist == "clusters":         # demo: dos cúmulos densos que chocan
    h = a.n // 2
    pos = np.vstack([rng.normal(0.3, 0.03, (h, 3)), rng.normal(0.7, 0.03, (a.n - h, 3))])
else:                              # demo: casi todo en un rincón
    k = int(a.n * 0.95)
    pos = np.vstack([rng.random((k, 3)) * 0.05, rng.random((a.n - k, 3))])
pos *= a.escala
vel = rng.normal(0.0, a.vel, (a.n, 3)) if a.vel > 0 else np.zeros((a.n, 3))
if a.dist == "clusters" and a.vel > 0:   # que se acerquen: uno a la derecha, otro a la izquierda
    vel[: a.n // 2, 0] += a.vel
    vel[a.n // 2 :, 0] -= a.vel
m = rng.random((a.n, 1)) if a.masas == "aleatorias" else np.ones((a.n, 1))
m /= m.sum()                       # masa total = 1
suf = ("_v" if a.vel > 0 else "") + ("_m" if a.masas == "aleatorias" else "") + ("_e" if a.escala != 1 else "")
np.savetxt(f"data/{a.dist}_N{a.n}_s{a.seed}{suf}.txt", np.hstack([pos, vel, m]))
```

Las distribuciones `clusters` y `corner` no las pide el enunciado: sirven para ensayar la demo, que usa una distribución desconocida "que castiga el reparto ingenuo".
