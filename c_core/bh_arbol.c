/* bh_arbol.c — caja, códigos de Morton, ordenamiento y construcción del octree aplanado.
 *
 * Por qué Morton: el código de una partícula intercala los bits de x, y, z. Los 3 bits
 * más altos dicen en qué octante de la raíz está; los 3 siguientes, en qué octante de
 * ese octante; y así. Al ordenar por código, todas las partículas de una misma celda
 * quedan JUNTAS en el arreglo. Construir el árbol se vuelve partir rangos, sin insertar
 * partícula por partícula. Esa misma propiedad es la que permite construir sin candados
 * en OpenMP (C2): cada subárbol escribe solo en su propio rango. */
#include <math.h>
#include <stdlib.h>
#include "bh_core.h"

/* ============================================================ caja */

/* Cubo que contiene a todas las partículas. Se recalcula cada paso porque se mueven,
 * y nunca se supone el cubo unitario: la demo puede traer posiciones fuera de [0, 1]. */
Caja calcular_caja(const Particulas *p) {
    double minx = p->x[0], maxx = p->x[0], miny = p->y[0], maxy = p->y[0];
    double minz = p->z[0], maxz = p->z[0];
    for (int i = 1; i < p->n; i++) {
        minx = fmin(minx, p->x[i]); maxx = fmax(maxx, p->x[i]);
        miny = fmin(miny, p->y[i]); maxy = fmax(maxy, p->y[i]);
        minz = fmin(minz, p->z[i]); maxz = fmax(maxz, p->z[i]);
    }
    double lado = fmax(maxx - minx, fmax(maxy - miny, maxz - minz));
    if (lado <= 0.0) lado = 1e-12;   /* todas en el mismo punto: evita dividir por cero */
    Caja c = {0.5 * (minx + maxx), 0.5 * (miny + maxy), 0.5 * (minz + maxz), 0.5 * lado};
    return c;
}

/* ============================================================ Morton */

/* Lleva una coordenada a un entero de 21 bits: 0 en el borde inferior de la caja,
 * 2^21 - 1 en el superior. El borde superior exacto daría 2^21: se recorta. */
static uint64_t a_entero(double coord, double centro, double mitad) {
    double t = (coord - (centro - mitad)) / (2.0 * mitad);   /* en [0, 1] */
    uint64_t maximo = ((uint64_t)1 << BITS_EJE) - 1;
    if (t <= 0.0) return 0;
    uint64_t v = (uint64_t)(t * (double)((uint64_t)1 << BITS_EJE));
    return v > maximo ? maximo : v;
}

/* Pone el bit b de v en la posición 3*b: ...b2 b1 b0 -> ..b2 0 0 b1 0 0 b0.
 * Un bucle y no "números mágicos": más lento, pero se explica en una línea. */
static uint64_t separar_bits(uint64_t v) {
    uint64_t r = 0;
    for (int b = 0; b < BITS_EJE; b++)
        r |= ((v >> b) & 1) << (3 * b);
    return r;
}

/* Código de 63 bits: x en los bits 3b+2, y en 3b+1, z en 3b. */
uint64_t codigo_morton(double x, double y, double z, Caja c) {
    uint64_t ix = a_entero(x, c.cx, c.mitad);
    uint64_t iy = a_entero(y, c.cy, c.mitad);
    uint64_t iz = a_entero(z, c.cz, c.mitad);
    return (separar_bits(ix) << 2) | (separar_bits(iy) << 1) | separar_bits(iz);
}

/* PARALELO (en C2 y C3): cada código depende solo de su partícula. */
void calcular_codigos(Particulas *p, Caja caja) {
    for (int i = 0; i < p->n; i++)
        p->codigo[i] = codigo_morton(p->x[i], p->y[i], p->z[i], caja);
}

/* Octante (0..7) de un código en un nivel: nivel 0 = hijos de la raíz = 3 bits más altos.
 * El octante es (bit x, bit y, bit z) = 4*bx + 2*by + bz. */
int octante(uint64_t codigo, int nivel) {
    int corrimiento = 3 * (BITS_EJE - 1 - nivel);
    return (int)((codigo >> corrimiento) & 7);
}

/* ============================================================ ordenamiento */

typedef struct { uint64_t codigo; int indice; } ParOrden;

static int comparar_pares(const void *a, const void *b) {
    const ParOrden *pa = a, *pb = b;
    if (pa->codigo != pb->codigo) return pa->codigo < pb->codigo ? -1 : 1;
    return pa->indice - pb->indice;   /* desempate fijo: mismo resultado en cada corrida */
}

/* destino[k] = origen[orden[k].indice] para un arreglo de doubles. */
static void permutar(double *arr, const ParOrden *orden, int n, double *tmp) {
    for (int k = 0; k < n; k++) tmp[k] = arr[orden[k].indice];
    for (int k = 0; k < n; k++) arr[k] = tmp[k];
}

/* Ordena las partículas por (código, índice) y mueve TODOS sus datos a ese orden.
 * id[] se mueve también: es lo que permite escribir las salidas en el orden original.
 * Las aceleraciones no se mueven: se recalculan justo después, con el árbol nuevo. */
void ordenar_por_morton(Particulas *p) {
    int n = p->n;
    ParOrden *orden = reservar((size_t)n * sizeof(ParOrden));
    for (int i = 0; i < n; i++) { orden[i].codigo = p->codigo[i]; orden[i].indice = i; }
    qsort(orden, n, sizeof(ParOrden), comparar_pares);
    double *tmp = reservar((size_t)n * sizeof(double));
    double *campos[] = {p->x, p->y, p->z, p->vx, p->vy, p->vz, p->m};
    for (int c = 0; c < 7; c++) permutar(campos[c], orden, n, tmp);
    int *id_tmp = reservar((size_t)n * sizeof(int));
    for (int k = 0; k < n; k++) id_tmp[k] = p->id[orden[k].indice];
    for (int k = 0; k < n; k++) { p->id[k] = id_tmp[k]; p->codigo[k] = orden[k].codigo; }
    free(id_tmp); free(tmp); free(orden);
}

/* ============================================================ árbol aplanado */

/* Deja espacio para al menos `cap` nodos en todos los arreglos del árbol. Secuencial:
 * en C2 se llama ANTES de repartir el trabajo, porque realloc entre hilos sería una
 * carrera (un hilo movería los arreglos mientras otro escribe en ellos). */
void reservar_nodos(Arbol *a, int cap) {
    if (cap <= a->capacidad) return;
    size_t d = (size_t)cap * sizeof(double), e = (size_t)cap * sizeof(int);
    a->hijos   = agrandar(a->hijos, 8 * e);
    a->inicio  = agrandar(a->inicio, e);  a->cuenta = agrandar(a->cuenta, e);
    a->es_hoja = agrandar(a->es_hoja, e); a->masa   = agrandar(a->masa, d);
    a->cmx = agrandar(a->cmx, d); a->cmy = agrandar(a->cmy, d); a->cmz = agrandar(a->cmz, d);
    a->cx  = agrandar(a->cx, d);  a->cy  = agrandar(a->cy, d);  a->cz  = agrandar(a->cz, d);
    a->mitad = agrandar(a->mitad, d);
    a->capacidad = cap;
}

/* Duplica la capacidad cuando el secuencial se queda sin espacio. */
static void crecer_arbol(Arbol *a) {
    reservar_nodos(a, a->capacidad < 64 ? 64 : 2 * a->capacidad);
}

/* Crea un nodo vacío para el rango [inicio, inicio+cuenta) y devuelve su índice. */
static int nuevo_nodo(Arbol *a, int inicio, int cuenta, double cx, double cy, double cz, double mitad) {
    if (a->n_nodos == a->capacidad) crecer_arbol(a);
    int k = a->n_nodos++;
    for (int o = 0; o < 8; o++) a->hijos[8 * k + o] = SIN_HIJO;
    a->inicio[k] = inicio; a->cuenta[k] = cuenta; a->es_hoja[k] = 0;
    a->cx[k] = cx; a->cy[k] = cy; a->cz[k] = cz; a->mitad[k] = mitad;
    return k;
}

/* Masa y centro de masa de una hoja: suma directa de sus partículas. */
void resumir_hoja(Arbol *a, int k, const Particulas *p) {
    double M = 0, sx = 0, sy = 0, sz = 0;
    for (int j = a->inicio[k]; j < a->inicio[k] + a->cuenta[k]; j++) {
        M += p->m[j]; sx += p->m[j] * p->x[j]; sy += p->m[j] * p->y[j]; sz += p->m[j] * p->z[j];
    }
    a->masa[k] = M; a->cmx[k] = sx / M; a->cmy[k] = sy / M; a->cmz[k] = sz / M;
}

/* Masa y centro de masa de un nodo interno: a partir de los resúmenes de sus hijos. */
void resumir_interno(Arbol *a, int k) {
    double M = 0, sx = 0, sy = 0, sz = 0;
    for (int o = 0; o < 8; o++) {
        int h = a->hijos[8 * k + o];
        if (h == SIN_HIJO) continue;
        M += a->masa[h]; sx += a->masa[h] * a->cmx[h];
        sy += a->masa[h] * a->cmy[h]; sz += a->masa[h] * a->cmz[h];
    }
    a->masa[k] = M; a->cmx[k] = sx / M; a->cmy[k] = sy / M; a->cmz[k] = sz / M;
}

static int construir_nodo(Arbol *a, const Particulas *p, int inicio, int cuenta, int nivel,
                          double cx, double cy, double cz, double mitad);

/* Avanza desde `desde` mientras las partículas sigan en el octante o: devuelve dónde termina.
 * Como están ordenadas por Morton, cada octante es un tramo contiguo. */
int fin_de_octante(const Particulas *p, int desde, int fin, int nivel, int o) {
    int j = desde;
    while (j < fin && octante(p->codigo[j], nivel) == o) j++;
    return j;
}

/* Parte el rango del nodo k en sus 8 octantes (un tramo contiguo cada uno). */
static void construir_hijos(Arbol *a, int k, const Particulas *p, int nivel) {
    int j = a->inicio[k], fin = a->inicio[k] + a->cuenta[k];
    double h = 0.5 * a->mitad[k];
    for (int o = 0; o < 8; o++) {
        int desde = j;
        j = fin_de_octante(p, desde, fin, nivel, o);
        if (j == desde) continue;                       /* octante vacío */
        double hx = a->cx[k] + ((o & 4) ? h : -h);      /* bit x del octante */
        double hy = a->cy[k] + ((o & 2) ? h : -h);      /* bit y */
        double hz = a->cz[k] + ((o & 1) ? h : -h);      /* bit z */
        /* PARALELO (en C2): cada llamada trabaja en un tramo distinto -> una task. */
        int hijo = construir_nodo(a, p, desde, j - desde, nivel + 1, hx, hy, hz, h);
        a->hijos[8 * k + o] = hijo;   /* después de la llamada: crecer_arbol pudo mover a->hijos */
    }
}

/* Crea el nodo del rango y, si tiene más de HOJA_MAX partículas, sus hijos.
 * En el nivel 21 ya no quedan bits: las partículas comparten código y se dejan en una
 * hoja aunque sean más de 8 (por eso el contrato habla de "hasta 8"). */
static int construir_nodo(Arbol *a, const Particulas *p, int inicio, int cuenta, int nivel,
                          double cx, double cy, double cz, double mitad) {
    int k = nuevo_nodo(a, inicio, cuenta, cx, cy, cz, mitad);
    if (cuenta <= HOJA_MAX || nivel == BITS_EJE) {
        a->es_hoja[k] = 1;
        resumir_hoja(a, k, p);
        return k;
    }
    construir_hijos(a, k, p, nivel);
    resumir_interno(a, k);
    return k;
}

/* Reconstruye el árbol completo. Requiere partículas ya ordenadas por Morton con la MISMA caja. */
void construir_arbol(Arbol *a, const Particulas *p, Caja caja) {
    a->n_nodos = 0;   /* se reutiliza la memoria del paso anterior */
    construir_nodo(a, p, 0, p->n, 0, caja.cx, caja.cy, caja.cz, caja.mitad);
}

void liberar_arbol(Arbol *a) {
    free(a->hijos); free(a->inicio); free(a->cuenta); free(a->es_hoja); free(a->masa);
    free(a->cmx); free(a->cmy); free(a->cmz); free(a->cx); free(a->cy); free(a->cz);
    free(a->mitad);
    a->n_nodos = a->capacidad = 0;
}
