/* bh_omp.c — Barnes-Hut con OpenMP (tarea C2). Usa el núcleo de c_core/.
 *
 *   make -C c_omp
 *   ./c_omp/bh_omp --input data/corner_N20000_s42.txt --steps 20 --dt 2.5e-4 --workers 4 --out /tmp/o
 *   OMP_SCHEDULE="static" ./c_omp/bh_omp ...        <- cambia el reparto de fuerzas SIN recompilar
 *   python common/verify.py correr --cmd "./c_omp/bh_omp" --workers 4
 *
 * Qué se paraleliza (marcas PARALELO) y qué queda secuencial:
 *   caja .............. parallel for con reduction(min/max)
 *   códigos Morton .... parallel for (cada código depende solo de su partícula)
 *   qsort + mover ..... SECUENCIAL (fracción secuencial de Amdahl; ver NOTAS.md)
 *   conteo de nodos ... SECUENCIAL y barato: reserva la memoria antes de repartir
 *   árbol ............. tasks recursivas, cortadas por CANTIDAD de partículas
 *   fuerzas ........... parallel for schedule(runtime): dynamic,64 por defecto
 *   integración ....... parallel for
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "../c_core/bh_core.h"

/* ------------------------------------------------------------ perillas de la demo */
/* Reparto de las fuerzas si no se define OMP_SCHEDULE. dynamic porque una partícula
 * dentro de un cúmulo abre muchas más celdas que una aislada (trabajo desigual). */
#define SCHEDULE_POR_DEFECTO  omp_sched_dynamic
#define CHUNK_POR_DEFECTO     64
/* Un nodo se vuelve task solo si tiene más de N / (FACTOR_CORTE · hilos) partículas.
 * Se corta por cantidad y no por nivel: con `corner` el 95% de las partículas cae en
 * UNA celda de nivel 2, y cortar por nivel dejaría casi todo el trabajo a un hilo. */
#define FACTOR_CORTE          8

/* Estadísticas por hilo para PREFIJO_workers.csv. Cada hilo escribe solo su casilla
 * [omp_get_thread_num()], así que no hay carrera. */
typedef struct {
    int     n;
    double *segundos;        /* tiempo ocupado en fuerzas */
    long   *interacciones;   /* trabajo real en fuerzas */
    long   *construidas;     /* partículas puestas en hojas por este hilo (reparto del árbol) */
} PorHilo;

/* ============================================================ configuración */

static PorHilo configurar_hilos(int workers) {
    omp_set_num_threads(workers);
    if (getenv("OMP_SCHEDULE") == NULL)                     /* respeta la variable si existe */
        omp_set_schedule(SCHEDULE_POR_DEFECTO, CHUNK_POR_DEFECTO);
    PorHilo h = {workers, NULL, NULL, NULL};
    h.segundos      = calloc(workers, sizeof(double));
    h.interacciones = calloc(workers, sizeof(long));
    h.construidas   = calloc(workers, sizeof(long));
    return h;
}

/* ============================================================ caja y códigos */

static Caja caja_paralela(const Particulas *p) {
    double minx = p->x[0], maxx = p->x[0], miny = p->y[0], maxy = p->y[0];
    double minz = p->z[0], maxz = p->z[0];
    /* PARALELO: cada hilo calcula su mínimo/máximo y reduction los combina al final. */
    #pragma omp parallel for reduction(min:minx,miny,minz) reduction(max:maxx,maxy,maxz)
    for (int i = 0; i < p->n; i++) {
        if (p->x[i] < minx) minx = p->x[i];
        if (p->x[i] > maxx) maxx = p->x[i];
        if (p->y[i] < miny) miny = p->y[i];
        if (p->y[i] > maxy) maxy = p->y[i];
        if (p->z[i] < minz) minz = p->z[i];
        if (p->z[i] > maxz) maxz = p->z[i];
    }
    double lado = maxx - minx;
    if (maxy - miny > lado) lado = maxy - miny;
    if (maxz - minz > lado) lado = maxz - minz;
    if (lado <= 0.0) lado = 1e-12;
    Caja c = {0.5 * (minx + maxx), 0.5 * (miny + maxy), 0.5 * (minz + maxz), 0.5 * lado};
    return c;
}

static void codigos_paralelos(Particulas *p, Caja caja) {
    /* PARALELO: cada código depende solo de su partícula. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < p->n; i++)
        p->codigo[i] = codigo_morton(p->x[i], p->y[i], p->z[i], caja);
}

/* ============================================================ árbol en paralelo */

/* Cuántos nodos tendrá el árbol (misma lógica que construir_nodo_omp, sin escribir nada).
 * Secuencial y barato: permite reservar TODA la memoria antes de crear tasks. */
static int contar_nodos(const Particulas *p, int inicio, int cuenta, int nivel) {
    if (cuenta <= HOJA_MAX || nivel == BITS_EJE) return 1;
    int total = 1, j = inicio, fin = inicio + cuenta;
    for (int o = 0; o < 8; o++) {
        int desde = j;
        j = fin_de_octante(p, desde, fin, nivel, o);
        if (j > desde) total += contar_nodos(p, desde, j - desde, nivel + 1);
    }
    return total;
}

/* Toma el siguiente índice libre. Varios hilos crean nodos a la vez: sin atomic, dos
 * podrían leer el mismo n_nodos y escribir en el mismo nodo. */
static int nuevo_nodo_omp(Arbol *a, int inicio, int cuenta, double cx, double cy, double cz, double mitad) {
    int k;
    /* SINCRONIZACIÓN: leer y aumentar n_nodos en una sola operación indivisible. */
    #pragma omp atomic capture
    k = a->n_nodos++;
    for (int o = 0; o < 8; o++) a->hijos[8 * k + o] = SIN_HIJO;
    a->inicio[k] = inicio; a->cuenta[k] = cuenta; a->es_hoja[k] = 0;
    a->cx[k] = cx; a->cy[k] = cy; a->cz[k] = cz; a->mitad[k] = mitad;
    return k;
}

static int construir_nodo_omp(Arbol *a, const Particulas *p, int inicio, int cuenta, int nivel,
                              double cx, double cy, double cz, double mitad, int corte, PorHilo *h);

/* Crea los hijos del nodo k. Cada hijo trabaja en un tramo DISTINTO de partículas y
 * escribe en una casilla distinta de hijos[], por eso no necesitan candados. */
static void hijos_omp(Arbol *a, int k, const Particulas *p, int nivel, int corte, PorHilo *h) {
    int j = a->inicio[k], fin = a->inicio[k] + a->cuenta[k];
    double m = 0.5 * a->mitad[k];
    for (int o = 0; o < 8; o++) {
        int desde = j;
        j = fin_de_octante(p, desde, fin, nivel, o);
        int n = j - desde;
        if (n == 0) continue;
        double hx = a->cx[k] + ((o & 4) ? m : -m);
        double hy = a->cy[k] + ((o & 2) ? m : -m);
        double hz = a->cz[k] + ((o & 1) ? m : -m);
        /* PARALELO: subárboles grandes se vuelven task; los pequeños se hacen aquí mismo,
         * porque crear una task cuesta más que construir unas pocas hojas. */
        #pragma omp task if(n > corte) firstprivate(o, desde, n, hx, hy, hz)
        a->hijos[8 * k + o] = construir_nodo_omp(a, p, desde, n, nivel + 1, hx, hy, hz, m, corte, h);
    }
    /* SINCRONIZACIÓN: la masa del nodo necesita la de TODOS sus hijos: esperar sus tasks. */
    #pragma omp taskwait
}

static int construir_nodo_omp(Arbol *a, const Particulas *p, int inicio, int cuenta, int nivel,
                              double cx, double cy, double cz, double mitad, int corte, PorHilo *h) {
    int k = nuevo_nodo_omp(a, inicio, cuenta, cx, cy, cz, mitad);
    if (cuenta <= HOJA_MAX || nivel == BITS_EJE) {
        a->es_hoja[k] = 1;
        resumir_hoja(a, k, p);
        h->construidas[omp_get_thread_num()] += cuenta;   /* casilla propia: sin carrera */
        return k;
    }
    hijos_omp(a, k, p, nivel, corte, h);
    resumir_interno(a, k);
    return k;
}

static void construir_arbol_omp(Arbol *a, const Particulas *p, Caja caja, PorHilo *h) {
    reservar_nodos(a, contar_nodos(p, 0, p->n, 0));   /* nada de realloc dentro de las tasks */
    a->n_nodos = 0;
    int corte = p->n / (FACTOR_CORTE * h->n);
    /* PARALELO: un hilo arranca la recursión (single) y los demás toman las tasks. */
    #pragma omp parallel
    #pragma omp single
    construir_nodo_omp(a, p, 0, p->n, 0, caja.cx, caja.cy, caja.cz, caja.mitad, corte, h);
}

/* ============================================================ fuerzas e integración */

static void fuerzas_omp(const Arbol *a, Particulas *p, double theta, double eps, PorHilo *h) {
    #pragma omp parallel
    {
        int yo = omp_get_thread_num();
        long mias = 0;
        double t0 = omp_get_wtime();
        /* PARALELO: cada i escribe solo ax[i], ay[i], az[i]; el árbol solo se lee.
         * schedule(runtime) lee OMP_SCHEDULE: static, dynamic,64, guided...
         * nowait: cada hilo mide cuándo terminó SU parte, así se ve el desbalance. */
        #pragma omp for schedule(runtime) nowait
        for (int i = 0; i < p->n; i++)
            mias += aceleracion_de(a, p, i, theta, eps, &p->ax[i], &p->ay[i], &p->az[i]);
        h->segundos[yo] += omp_get_wtime() - t0;
        h->interacciones[yo] += mias;
    }   /* SINCRONIZACIÓN: barrera implícita al cerrar parallel: todas las fuerzas listas. */
}

static void medio_kick_omp(Particulas *p, double dt) {
    /* PARALELO: cada partícula se actualiza sola. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < p->n; i++) {
        p->vx[i] += 0.5 * dt * p->ax[i]; p->vy[i] += 0.5 * dt * p->ay[i]; p->vz[i] += 0.5 * dt * p->az[i];
    }
}

static void drift_omp(Particulas *p, double dt) {
    /* PARALELO: cada partícula se actualiza sola. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < p->n; i++) {
        p->x[i] += dt * p->vx[i]; p->y[i] += dt * p->vy[i]; p->z[i] += dt * p->vz[i];
    }
}

/* ============================================================ un paso */

static void calcular_fuerzas(Particulas *p, Arbol *arbol, const Parametros *par, Tiempos *t, PorHilo *h) {
    double t0 = omp_get_wtime();
    Caja caja = caja_paralela(p);
    codigos_paralelos(p, caja);
    ordenar_por_morton(p);                      /* secuencial, de c_core/ */
    double t1 = omp_get_wtime();
    construir_arbol_omp(arbol, p, caja, h);
    double t2 = omp_get_wtime();
    fuerzas_omp(arbol, p, par->theta, par->eps, h);
    double t3 = omp_get_wtime();
    t->ordenamiento += t1 - t0;
    t->construccion += t2 - t1;
    t->recorrido    += t3 - t2;
}

static void paso(Particulas *p, Arbol *arbol, const Parametros *par, Tiempos *t, PorHilo *h) {
    double t0 = omp_get_wtime();
    medio_kick_omp(p, par->dt);
    drift_omp(p, par->dt);
    t->integracion += omp_get_wtime() - t0;
    calcular_fuerzas(p, arbol, par, t, h);
    t0 = omp_get_wtime();
    medio_kick_omp(p, par->dt);
    t->integracion += omp_get_wtime() - t0;
}

/* PREFIJO_workers.csv: un hilo por fila. Si los segundos o las interacciones difieren
 * mucho entre hilos, el reparto está desbalanceado. */
static void escribir_hilos(const char *prefijo, const PorHilo *h) {
    char ruta[4096];
    snprintf(ruta, sizeof ruta, "%s_workers.csv", prefijo);
    FILE *f = fopen(ruta, "w");
    if (!f) { fprintf(stderr, "no se puede escribir %s\n", ruta); exit(1); }
    fprintf(f, "hilo,segundos,interacciones,particulas_construidas\n");
    for (int k = 0; k < h->n; k++)
        fprintf(f, "%d,%.6f,%ld,%ld\n", k, h->segundos[k], h->interacciones[k], h->construidas[k]);
    fclose(f);
}

int main(int argc, char **argv) {
    Parametros par = leer_parametros(argc, argv);
    PorHilo h = configurar_hilos(par.workers);
    Tiempos t = {0};
    Arbol arbol = {0};
    double inicio = omp_get_wtime();

    Particulas p = leer_particulas(par.entrada);
    t.lectura = omp_get_wtime() - inicio;

    calcular_fuerzas(&p, &arbol, &par, &t, &h);   /* aceleraciones del paso 0 */
    escribir_aceleraciones(par.salida, &p);
    for (int s = 0; s < par.pasos; s++)
        paso(&p, &arbol, &par, &t, &h);

    t.total = omp_get_wtime() - inicio;          /* comunicacion = 0: memoria compartida */
    escribir_estado_final(par.salida, &p);
    escribir_tiempos(par.salida, &t);
    escribir_hilos(par.salida, &h);
    printf("N = %d, pasos = %d, hilos = %d, theta = %g, nodos = %d, total = %.3f s\n",
           p.n, par.pasos, par.workers, par.theta, arbol.n_nodos, t.total);
    liberar_arbol(&arbol);
    liberar_particulas(&p);
    return 0;
}
