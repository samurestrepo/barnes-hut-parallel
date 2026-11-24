"""Física de referencia por suma directa O(N²), tal como la fija CONTRATO.md.

La usan verify.py y directo.py. Es lenta a propósito: es la verdad contra la
que se compara Barnes-Hut, así que prima que sea obviamente correcta.
"""
import numpy as np

G = 1.0

# Cuántos pares (i, j) se calculan a la vez. Con N = 20 000 la matriz completa
# de diferencias ocuparía ~10 GB; por bloques de filas cabe en ~25 MB.
PARES_POR_BLOQUE = 1_000_000


def leer_particulas(ruta):
    """Lee un archivo N×7 (x y z vx vy vz m) y devuelve pos, vel, m."""
    datos = np.loadtxt(ruta, ndmin=2)
    if datos.shape[1] != 7:
        raise ValueError(f"{ruta}: se esperaban 7 columnas, hay {datos.shape[1]}")
    return datos[:, 0:3].copy(), datos[:, 3:6].copy(), datos[:, 6].copy()


def bloques_de_filas(n):
    """Rangos [i0, i1) de filas tales que cada bloque tenga ~PARES_POR_BLOQUE pares."""
    paso = max(1, PARES_POR_BLOQUE // n)
    for i0 in range(0, n, paso):
        yield i0, min(n, i0 + paso)


def diferencias_y_distancia2(pos, i0, i1, eps):
    """d[b, j] = r_j - r_i (i = i0 + b) y |d|² + ε² para un bloque de filas."""
    d = pos[None, :, :] - pos[i0:i1, None, :]
    r2 = np.einsum("bjk,bjk->bj", d, d) + eps * eps
    return d, r2


def anular_diagonal(matriz, i0):
    """Pone en 0 los términos j = i del bloque: una partícula no se atrae a sí misma."""
    filas = np.arange(matriz.shape[0])
    matriz[filas, i0 + filas] = 0.0


def aceleraciones_directas(pos, m, eps):
    """a_i = Σ_{j≠i} G m_j (r_j - r_i) / (|r_j - r_i|² + ε²)^{3/2}."""
    acc = np.empty_like(pos)
    for i0, i1 in bloques_de_filas(len(pos)):
        d, r2 = diferencias_y_distancia2(pos, i0, i1, eps)
        with np.errstate(divide="ignore", invalid="ignore"):  # j = i con ε = 0: se anula abajo
            peso = m[None, :] / (r2 * np.sqrt(r2))
        anular_diagonal(peso, i0)
        acc[i0:i1] = G * np.einsum("bj,bjk->bk", peso, d)
    return acc


def energia_total(pos, vel, m, eps):
    """E = Σ ½ m v² − Σ_{i<j} G m_i m_j / sqrt(|r_i − r_j|² + ε²)."""
    cinetica = 0.5 * np.sum(m * np.einsum("ik,ik->i", vel, vel))
    suma_pares = 0.0
    for i0, i1 in bloques_de_filas(len(pos)):
        _, r2 = diferencias_y_distancia2(pos, i0, i1, eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            termino = m[i0:i1, None] * m[None, :] / np.sqrt(r2)
        anular_diagonal(termino, i0)
        suma_pares += termino.sum()
    # Cada par aparece dos veces, como (i, j) y como (j, i): por eso la mitad.
    return cinetica - G * suma_pares / 2.0


def leapfrog_directo(pos, vel, m, dt, eps, pasos):
    """Integra `pasos` pasos kick-drift-kick con suma directa. No modifica la entrada."""
    pos, vel = pos.copy(), vel.copy()
    acc = aceleraciones_directas(pos, m, eps)
    for _ in range(pasos):
        vel += 0.5 * dt * acc   # medio kick con las aceleraciones viejas
        pos += dt * vel         # drift completo
        acc = aceleraciones_directas(pos, m, eps)
        vel += 0.5 * dt * acc   # medio kick con las nuevas: sin él, la energía deriva
    return pos, vel
