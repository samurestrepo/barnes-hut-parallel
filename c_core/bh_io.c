/* bh_io.c — todo lo que no es física: CLI, lectura, escritura, reloj y memoria.
 * Aquí no hay concurrencia: se ejecuta antes y después de la simulación. */
#define _POSIX_C_SOURCE 199309L   /* para clock_gettime */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bh_core.h"

/* 17 cifras significativas: el texto no debe sumar error de redondeo a float64
 * (con %.6e el verificador fallaba por redondeo; ver CONTRATO_DUDAS.md). */
#define FORMATO "%.17g"

static void morir(const char *mensaje, const char *detalle) {
    fprintf(stderr, "error: %s %s\n", mensaje, detalle ? detalle : "");
    exit(1);
}

/* malloc que nunca devuelve NULL: si falta memoria, el programa se detiene con un mensaje. */
void *reservar(size_t bytes) {
    void *ptr = malloc(bytes);
    if (ptr == NULL && bytes > 0) morir("sin memoria", NULL);
    return ptr;
}

void *agrandar(void *ptr, size_t bytes) {
    ptr = realloc(ptr, bytes);
    if (ptr == NULL) morir("sin memoria", NULL);
    return ptr;
}

/* Segundos desde un punto fijo; solo sirve para restar dos lecturas. */
double reloj(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* CLI del contrato: --input F --steps S --dt --theta --eps --workers K --out PREFIJO */
Parametros leer_parametros(int argc, char **argv) {
    Parametros par = {NULL, NULL, -1, 0.001, 0.5, 0.01, 1, 0};
    for (int k = 1; k < argc; k++) {
        const char *op = argv[k];
        if (strcmp(op, "--directa") == 0) { par.directa = 1; continue; }
        if (k + 1 >= argc) morir("falta el valor de", op);
        const char *valor = argv[++k];
        if      (strcmp(op, "--input")   == 0) par.entrada = valor;
        else if (strcmp(op, "--out")     == 0) par.salida  = valor;
        else if (strcmp(op, "--steps")   == 0) par.pasos   = atoi(valor);
        else if (strcmp(op, "--dt")      == 0) par.dt      = atof(valor);
        else if (strcmp(op, "--theta")   == 0) par.theta   = atof(valor);
        else if (strcmp(op, "--eps")     == 0) par.eps     = atof(valor);
        else if (strcmp(op, "--workers") == 0) par.workers = atoi(valor);
        else morir("opción desconocida", op);
    }
    if (!par.entrada || !par.salida || par.pasos < 0)
        morir("uso: --input F --steps S --out PREFIJO [--dt --theta --eps --workers --directa]", NULL);
    return par;
}

/* Reserva todos los arreglos de n partículas. */
static Particulas reservar_particulas(int n) {
    Particulas p;
    size_t d = (size_t)n * sizeof(double);
    p.n = n;
    p.x  = reservar(d); p.y  = reservar(d); p.z  = reservar(d);
    p.vx = reservar(d); p.vy = reservar(d); p.vz = reservar(d);
    p.ax = reservar(d); p.ay = reservar(d); p.az = reservar(d);
    p.m  = reservar(d);
    p.id = reservar((size_t)n * sizeof(int));
    p.codigo = reservar((size_t)n * sizeof(uint64_t));
    return p;
}

/* Cuenta las filas para saber N antes de reservar (el archivo no trae encabezado). */
static int contar_filas(FILE *f) {
    double v[7];
    int n = 0;
    while (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) == 7)
        n++;
    rewind(f);
    return n;
}

/* Lee el archivo N×7 (x y z vx vy vz m). id[i] = i guarda el orden de la entrada. */
Particulas leer_particulas(const char *ruta) {
    FILE *f = fopen(ruta, "r");
    if (!f) morir("no se puede abrir", ruta);
    int n = contar_filas(f);
    if (n == 0) morir("archivo vacío o con formato distinto a N×7:", ruta);
    Particulas p = reservar_particulas(n);
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf", &p.x[i], &p.y[i], &p.z[i],
                   &p.vx[i], &p.vy[i], &p.vz[i], &p.m[i]) != 7)
            morir("fila mal formada en", ruta);
        p.id[i] = i;
    }
    fclose(f);
    return p;
}

static FILE *abrir_salida(const char *prefijo, const char *sufijo) {
    char ruta[4096];
    snprintf(ruta, sizeof ruta, "%s%s", prefijo, sufijo);
    FILE *f = fopen(ruta, "w");
    if (!f) morir("no se puede escribir", ruta);
    return f;
}

/* Las partículas están en orden Morton: orden[id[i]] = i las devuelve al orden de la entrada.
 * Si se escribieran en orden Morton, la prueba "trayectoria" del verificador fallaría. */
static int *orden_original(const Particulas *p) {
    int *orden = reservar((size_t)p->n * sizeof(int));
    for (int i = 0; i < p->n; i++) orden[p->id[i]] = i;
    return orden;
}

void escribir_aceleraciones(const char *prefijo, const Particulas *p) {
    FILE *f = abrir_salida(prefijo, "_acc0.txt");
    int *orden = orden_original(p);
    for (int k = 0; k < p->n; k++) {
        int i = orden[k];
        fprintf(f, FORMATO " " FORMATO " " FORMATO "\n", p->ax[i], p->ay[i], p->az[i]);
    }
    free(orden);
    fclose(f);
}

void escribir_estado_final(const char *prefijo, const Particulas *p) {
    FILE *f = abrir_salida(prefijo, "_final.txt");
    int *orden = orden_original(p);
    for (int k = 0; k < p->n; k++) {
        int i = orden[k];
        fprintf(f, FORMATO " " FORMATO " " FORMATO " " FORMATO " " FORMATO " " FORMATO " " FORMATO "\n",
                p->x[i], p->y[i], p->z[i], p->vx[i], p->vy[i], p->vz[i], p->m[i]);
    }
    free(orden);
    fclose(f);
}

void escribir_tiempos(const char *prefijo, const Tiempos *t) {
    FILE *f = abrir_salida(prefijo, "_times.csv");
    fprintf(f, "fase,segundos\n");
    fprintf(f, "lectura,%.6f\nconstruccion,%.6f\nordenamiento,%.6f\n", t->lectura, t->construccion, t->ordenamiento);
    fprintf(f, "recorrido,%.6f\nintegracion,%.6f\n", t->recorrido, t->integracion);
    fprintf(f, "comunicacion,%.6f\ntotal,%.6f\n", t->comunicacion, t->total);
    fclose(f);
}

void liberar_particulas(Particulas *p) {
    free(p->x);  free(p->y);  free(p->z);
    free(p->vx); free(p->vy); free(p->vz);
    free(p->ax); free(p->ay); free(p->az);
    free(p->m);  free(p->id); free(p->codigo);
}
