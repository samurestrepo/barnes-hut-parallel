/* bh_fuerzas.c — recorrido del árbol (aceleraciones), suma directa e integrador leapfrog.
 *
 *   a_i = Σ_j G m_j (r_j - r_i) / (|r_j - r_i|² + ε²)^{3/2}       (CONTRATO.md)
 *
 * Criterio de apertura: una celda de lado L a distancia d del centro de masa se trata
 * como UNA partícula si L / d < θ. Con θ = 0 nunca se aproxima: es la suma directa. */
#include <math.h>
#include "bh_core.h"

/* Suma a (ax, ay, az) la atracción de una masa mj desplazada (dx, dy, dz) desde la partícula. */
static void sumar_atraccion(double dx, double dy, double dz, double mj, double eps2,
                            double *ax, double *ay, double *az) {
    double r2 = dx * dx + dy * dy + dz * dz + eps2;
    double factor = G_GRAV * mj / (r2 * sqrt(r2));
    *ax += factor * dx; *ay += factor * dy; *az += factor * dz;
}

/* 1 si el punto está dentro de la caja de la celda k. Esa celda se abre SIEMPRE:
 * su centro de masa incluye a la propia partícula, y con θ grande (θ ≳ 0.58, la demo
 * puede pedir θ = 1) la partícula se atraería a sí misma. */
static int celda_contiene(const Arbol *a, int k, double x, double y, double z) {
    return fabs(x - a->cx[k]) <= a->mitad[k] && fabs(y - a->cy[k]) <= a->mitad[k]
        && fabs(z - a->cz[k]) <= a->mitad[k];
}

/* Hoja: se suman sus partículas una por una, saltándose a la propia i. */
static void sumar_hoja(const Arbol *a, int k, const Particulas *p, int i, double eps2,
                       double *ax, double *ay, double *az) {
    for (int j = a->inicio[k]; j < a->inicio[k] + a->cuenta[k]; j++) {
        if (j == i) continue;
        sumar_atraccion(p->x[j] - p->x[i], p->y[j] - p->y[i], p->z[j] - p->z[i], p->m[j], eps2, ax, ay, az);
    }
}

/* Aceleración de la partícula i recorriendo el árbol SIN recursión, con una pila explícita.
 * Es la misma forma que tendrá el kernel de CUDA (C3), donde la recursión es cara. */
void aceleracion_de(const Arbol *a, const Particulas *p, int i, double theta, double eps,
                    double *ax, double *ay, double *az) {
    int pila[PILA_MAX], tope = 0;
    double eps2 = eps * eps, xi = p->x[i], yi = p->y[i], zi = p->z[i];
    *ax = *ay = *az = 0.0;
    pila[tope++] = 0;                                       /* la raíz */
    while (tope > 0) {
        int k = pila[--tope];
        if (a->es_hoja[k]) { sumar_hoja(a, k, p, i, eps2, ax, ay, az); continue; }
        double dx = a->cmx[k] - xi, dy = a->cmy[k] - yi, dz = a->cmz[k] - zi;
        double distancia = sqrt(dx * dx + dy * dy + dz * dz);
        double lado = 2.0 * a->mitad[k];
        if (!celda_contiene(a, k, xi, yi, zi) && lado < theta * distancia) {
            sumar_atraccion(dx, dy, dz, a->masa[k], eps2, ax, ay, az);   /* lejos: una sola masa */
            continue;
        }
        for (int o = 0; o < 8; o++) {                       /* cerca: se abre la celda */
            int h = a->hijos[8 * k + o];
            if (h != SIN_HIJO) pila[tope++] = h;
        }
    }
}

/* PARALELO (en C2 con schedule(runtime), en C3 un hilo de GPU por partícula):
 * cada i lee el árbol y escribe solo su propia aceleración, así que no hay carrera. */
void calcular_aceleraciones(const Arbol *a, Particulas *p, double theta, double eps) {
    for (int i = 0; i < p->n; i++)
        aceleracion_de(a, p, i, theta, eps, &p->ax[i], &p->ay[i], &p->az[i]);
}

/* O(N²): solo para depurar (--directa). Debe coincidir con el árbol cuando θ = 0. */
void suma_directa(Particulas *p, double eps) {
    double eps2 = eps * eps;
    for (int i = 0; i < p->n; i++) {
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < p->n; j++) {
            if (j == i) continue;
            sumar_atraccion(p->x[j] - p->x[i], p->y[j] - p->y[i], p->z[j] - p->z[i], p->m[j], eps2, &ax, &ay, &az);
        }
        p->ax[i] = ax; p->ay[i] = ay; p->az[i] = az;
    }
}

/* Leapfrog kick-drift-kick: medio_kick -> drift -> recalcular aceleraciones -> medio_kick.
 * Sin el segundo medio kick la energía deriva (el verificador lo atrapa). */
void medio_kick(Particulas *p, double dt) {
    for (int i = 0; i < p->n; i++) {
        p->vx[i] += 0.5 * dt * p->ax[i]; p->vy[i] += 0.5 * dt * p->ay[i]; p->vz[i] += 0.5 * dt * p->az[i];
    }
}

void drift(Particulas *p, double dt) {
    for (int i = 0; i < p->n; i++) {
        p->x[i] += dt * p->vx[i]; p->y[i] += dt * p->vy[i]; p->z[i] += dt * p->vz[i];
    }
}
