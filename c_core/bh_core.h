/* bh_core.h — núcleo de Barnes-Hut en C (tarea C1).
 *
 * Lo usan las tres versiones en C de Sebastián:
 *   c_core/bh.c        secuencial (este núcleo + un main)
 *   c_omp/bh_omp.c     OpenMP: reemplaza las fases calientes por versiones paralelas
 *   cuda/bh_cuda.cu    CUDA: construye el árbol aquí (CPU) y recorre en la GPU
 *
 * Idea general de un paso:
 *   1. caja     : cubo que contiene a todas las partículas (se recalcula cada paso).
 *   2. morton   : a cada partícula un código de 63 bits que intercala x, y, z.
 *   3. orden    : se ordenan las partículas por código. Así cada celda del octree
 *                 es un RANGO CONTIGUO [inicio, inicio+cuenta) del arreglo.
 *   4. árbol    : octree APLANADO en arreglos (nada de punteros a structs).
 *   5. recorrido: aceleración de cada partícula con una pila explícita (sin recursión).
 *   6. leapfrog : kick-drift-kick.
 */
#ifndef BH_CORE_H
#define BH_CORE_H

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------ perillas fijas */
#define G_GRAV     1.0   /* constante gravitacional del contrato */
#define HOJA_MAX   8     /* hasta 8 partículas por hoja (CONTRATO.md) */
#define BITS_EJE   21    /* bits de Morton por eje: 3*21 = 63 bits caben en uint64 */
#define PILA_MAX   256   /* profundidad <= 21 niveles y cada nivel apila <= 7 hermanos */
#define SIN_HIJO   (-1)  /* marca de "este hijo no existe" en arbol.hijos */

/* ------------------------------------------------------------ parámetros (CLI) */
typedef struct {
    const char *entrada;   /* --input   archivo N×7 */
    const char *salida;    /* --out     PREFIJO de las salidas */
    int    pasos;          /* --steps */
    double dt;             /* --dt      0.001 uniform; 2.5e-4 concentradas */
    double theta;          /* --theta   0 = suma directa; 0.5 por defecto; la demo puede pedir 1.0 */
    double eps;            /* --eps     suavizado */
    int    workers;        /* --workers hilos (OpenMP) o hilos por bloque (CUDA); el secuencial lo ignora */
    int    directa;        /* --directa usar suma_directa() en vez del árbol (solo para depurar) */
} Parametros;

/* ------------------------------------------------------------ partículas
 * "Estructura de arreglos" (SoA): x[i], y[i]... en vez de un struct por partícula.
 * Así los hilos de OpenMP y de CUDA leen memoria contigua. */
typedef struct {
    int     n;
    double *x, *y, *z;       /* posición */
    double *vx, *vy, *vz;    /* velocidad */
    double *ax, *ay, *az;    /* aceleración */
    double *m;               /* masa */
    int    *id;              /* posición en el archivo de entrada: para escribir en el orden original */
    uint64_t *codigo;        /* código de Morton del paso actual */
} Particulas;

/* ------------------------------------------------------------ caja raíz (siempre un cubo) */
typedef struct {
    double cx, cy, cz;   /* centro */
    double mitad;        /* mitad del lado */
} Caja;

/* ------------------------------------------------------------ octree aplanado
 * El nodo k tiene sus datos en la posición k de cada arreglo.
 * Sus 8 hijos están en hijos[8*k + 0..7] (SIN_HIJO si ese octante está vacío).
 * La raíz es siempre el nodo 0. */
typedef struct {
    int     n_nodos, capacidad;
    int    *hijos;              /* 8 por nodo */
    int    *inicio, *cuenta;    /* rango de partículas ordenadas que cubre el nodo */
    int    *es_hoja;            /* 1 si es hoja: sus partículas se suman una por una */
    double *masa;               /* masa total de la celda */
    double *cmx, *cmy, *cmz;    /* centro de masa */
    double *cx, *cy, *cz;       /* centro geométrico de la celda */
    double *mitad;              /* mitad del lado de la celda */
} Arbol;

/* ------------------------------------------------------------ tiempos por fase (CONTRATO.md) */
typedef struct {
    double lectura, construccion, ordenamiento, recorrido, integracion, comunicacion, total;
} Tiempos;

/* entrada / salida */
Parametros leer_parametros(int argc, char **argv);
Particulas leer_particulas(const char *ruta);
void escribir_aceleraciones(const char *prefijo, const Particulas *p);
void escribir_estado_final(const char *prefijo, const Particulas *p);
void escribir_tiempos(const char *prefijo, const Tiempos *t);
void liberar_particulas(Particulas *p);
double reloj(void);
void  *reservar(size_t bytes);              /* malloc que se detiene si falta memoria */
void  *agrandar(void *ptr, size_t bytes);   /* realloc que se detiene si falta memoria */

/* fases de un paso */
Caja     calcular_caja(const Particulas *p);
uint64_t codigo_morton(double x, double y, double z, Caja caja);
void     calcular_codigos(Particulas *p, Caja caja);
void     ordenar_por_morton(Particulas *p);
void     construir_arbol(Arbol *a, const Particulas *p, Caja caja);
void     aceleracion_de(const Arbol *a, const Particulas *p, int i, double theta, double eps,
                        double *ax, double *ay, double *az);
void     calcular_aceleraciones(const Arbol *a, Particulas *p, double theta, double eps);
void     suma_directa(Particulas *p, double eps);
void     medio_kick(Particulas *p, double dt);
void     drift(Particulas *p, double dt);
void     liberar_arbol(Arbol *a);

#endif
