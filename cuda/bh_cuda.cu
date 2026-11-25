/* bh_cuda.cu — Barnes-Hut en CUDA (tarea C3). Se compila y corre en Colab con una T4.
 *
 *   make -C cuda                                   (compila c_core/ con gcc y esto con nvcc)
 *   ./cuda/bh_cuda --input data/uniform_N20000_s42.txt --steps 20 --workers 128 --out /tmp/g
 *   ORDEN_HILOS=original ./cuda/bh_cuda ...        <- cambia la asignación a warps SIN recompilar
 *   python common/verify.py correr --cmd "./cuda/bh_cuda" --float32 --workers 128
 *
 * Un paso en la GPU:
 *   ordenamiento : caja (thrust::minmax) -> códigos Morton (kernel) -> thrust::sort_by_key
 *                  -> mover las partículas a ese orden (thrust::gather)
 *   construccion : copiar posiciones GPU->CPU, armar el octree con c_core/ en la CPU,
 *                  aplanarlo a float y copiarlo CPU->GPU (todo eso se mide aquí)
 *   recorrido    : un hilo por partícula, pila en memoria local, SIN recursión
 *   integracion  : kernels de medio kick y drift
 * --workers = hilos por bloque. Se calcula en float32: la T4 es mucho más rápida en FP32
 * que en FP64 (por eso el verificador usa --float32).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
extern "C" {
#include "../c_core/bh_core.h"
}

/* ------------------------------------------------------------ perillas de la demo */
/* Qué partícula atiende el hilo t:
 *   MORTON   -> la t-ésima en orden Morton: los 32 hilos de un warp son vecinos en el
 *               espacio, abren casi las mismas celdas y casi no divergen.
 *   ORIGINAL -> la de id t en el archivo: un warp mezcla partículas de todo el espacio,
 *               cada hilo recorre el árbol distinto y el warp espera al más lento.
 * La variable de entorno ORDEN_HILOS=morton|original la cambia sin recompilar. */
enum { MORTON = 0, ORIGINAL = 1 };
#define ORDEN_HILOS_POR_DEFECTO MORTON

/* Detiene el programa con el archivo y la línea si una llamada de CUDA falla. */
#define REVISAR(llamada) do { cudaError_t e_ = (llamada); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s en %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

/* Partículas en la GPU (float32, en orden Morton después de cada ordenamiento). */
struct ParticulasGPU {
    int n;
    float *x, *y, *z, *vx, *vy, *vz, *ax, *ay, *az, *m;
    int *id;                 /* posición en el archivo de entrada */
    unsigned long long *codigo;
    int *indice;             /* permutación que devuelve sort_by_key */
    int *hilo_a_particula;   /* ORIGINAL: hilo t -> posición actual de la partícula de id t */
    float *tmp; int *tmp_int;
};

/* Octree aplanado en la GPU: mismos arreglos que Arbol de c_core/, en float. */
struct ArbolGPU {
    int capacidad;
    int *hijos, *inicio, *cuenta, *es_hoja;
    float *masa, *cmx, *cmy, *cmz, *cx, *cy, *cz, *mitad;
};

/* Reloj de la GPU: mide entre dos eventos del mismo flujo, sin sincronizar a mano cada kernel. */
struct Cronometro { cudaEvent_t ini, fin; };

static void empezar(Cronometro *c) { REVISAR(cudaEventRecord(c->ini)); }
static double terminar(Cronometro *c) {
    float ms;
    REVISAR(cudaEventRecord(c->fin));
    REVISAR(cudaEventSynchronize(c->fin));
    REVISAR(cudaEventElapsedTime(&ms, c->ini, c->fin));
    return ms / 1000.0;
}

/* ============================================================ kernels */

/* Misma cuenta que codigo_morton() de c_core/, en double para que el borde de la celda
 * coincida con el que usa el árbol (que se arma en la CPU en double). */
__device__ unsigned long long separar_bits_gpu(unsigned long long v) {
    unsigned long long r = 0;
    for (int b = 0; b < BITS_EJE; b++) r |= ((v >> b) & 1ULL) << (3 * b);
    return r;
}

__device__ unsigned long long a_entero_gpu(double coord, double centro, double mitad) {
    double t = (coord - (centro - mitad)) / (2.0 * mitad);
    unsigned long long maximo = (1ULL << BITS_EJE) - 1;
    if (t <= 0.0) return 0;
    unsigned long long v = (unsigned long long)(t * (double)(1ULL << BITS_EJE));
    return v > maximo ? maximo : v;
}

/* PARALELO: un hilo por partícula; cada código depende solo de su partícula. */
__global__ void kernel_codigos(ParticulasGPU g, Caja c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n) return;
    unsigned long long ix = a_entero_gpu(g.x[i], c.cx, c.mitad);
    unsigned long long iy = a_entero_gpu(g.y[i], c.cy, c.mitad);
    unsigned long long iz = a_entero_gpu(g.z[i], c.cz, c.mitad);
    g.codigo[i] = (separar_bits_gpu(ix) << 2) | (separar_bits_gpu(iy) << 1) | separar_bits_gpu(iz);
    g.indice[i] = i;
}

/* hilo_a_particula[id[k]] = k: dónde quedó, tras ordenar, la partícula de cada id.
 * Sin carrera: los id son una permutación, cada hilo escribe una casilla distinta. */
__global__ void kernel_hilo_a_particula(ParticulasGPU g) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < g.n) g.hilo_a_particula[g.id[k]] = k;
}

__device__ void sumar_atraccion_gpu(float dx, float dy, float dz, float mj, float eps2,
                                    float *ax, float *ay, float *az) {
    float r2 = dx * dx + dy * dy + dz * dz + eps2;
    float inv = rsqrtf(r2);                       /* 1/sqrt en una instrucción */
    float f = G_GRAV * mj * inv * inv * inv;
    *ax += f * dx; *ay += f * dy; *az += f * dz;
}

__device__ bool celda_contiene_gpu(ArbolGPU a, int k, float x, float y, float z) {
    return fabsf(x - a.cx[k]) <= a.mitad[k] && fabsf(y - a.cy[k]) <= a.mitad[k]
        && fabsf(z - a.cz[k]) <= a.mitad[k];
}

__device__ void sumar_hoja_gpu(ArbolGPU a, ParticulasGPU g, int k, int i, float eps2,
                               float *ax, float *ay, float *az) {
    for (int j = a.inicio[k]; j < a.inicio[k] + a.cuenta[k]; j++)
        if (j != i)
            sumar_atraccion_gpu(g.x[j] - g.x[i], g.y[j] - g.y[i], g.z[j] - g.z[i], g.m[j], eps2, ax, ay, az);
}

/* PARALELO: un hilo por partícula. Es el mismo recorrido que aceleracion_de() de c_core/:
 * pila explícita (la recursión en GPU usa mucha pila y es lenta), la celda que contiene
 * a la partícula se abre siempre. La pila vive en memoria LOCAL de cada hilo: no se
 * comparte, así que no hay carrera ni hace falta __syncthreads(). */
__global__ void kernel_recorrido(ArbolGPU a, ParticulasGPU g, float theta, float eps, int orden) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= g.n) return;
    int i = (orden == MORTON) ? t : g.hilo_a_particula[t];
    float xi = g.x[i], yi = g.y[i], zi = g.z[i], eps2 = eps * eps, ax = 0, ay = 0, az = 0;
    int pila[PILA_MAX], tope = 0;
    pila[tope++] = 0;
    while (tope > 0) {
        int k = pila[--tope];
        if (a.es_hoja[k]) { sumar_hoja_gpu(a, g, k, i, eps2, &ax, &ay, &az); continue; }
        float dx = a.cmx[k] - xi, dy = a.cmy[k] - yi, dz = a.cmz[k] - zi;
        float distancia = sqrtf(dx * dx + dy * dy + dz * dz);
        if (!celda_contiene_gpu(a, k, xi, yi, zi) && 2.0f * a.mitad[k] < theta * distancia) {
            sumar_atraccion_gpu(dx, dy, dz, a.masa[k], eps2, &ax, &ay, &az);
            continue;
        }
        for (int o = 0; o < 8; o++) {
            int h = a.hijos[8 * k + o];
            if (h != SIN_HIJO) pila[tope++] = h;
        }
    }
    g.ax[i] = ax; g.ay[i] = ay; g.az[i] = az;    /* cada hilo escribe solo su partícula */
}

/* PARALELO: integración, un hilo por partícula. */
__global__ void kernel_medio_kick(ParticulasGPU g, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n) return;
    g.vx[i] += 0.5f * dt * g.ax[i]; g.vy[i] += 0.5f * dt * g.ay[i]; g.vz[i] += 0.5f * dt * g.az[i];
}

__global__ void kernel_drift(ParticulasGPU g, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n) return;
    g.x[i] += dt * g.vx[i]; g.y[i] += dt * g.vy[i]; g.z[i] += dt * g.vz[i];
}

/* ============================================================ memoria y copias */

static float *subir_floats(const double *h, int n, float *buf) {
    float *d;
    for (int i = 0; i < n; i++) buf[i] = (float)h[i];
    REVISAR(cudaMalloc(&d, n * sizeof(float)));
    REVISAR(cudaMemcpy(d, buf, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static void bajar_floats(double *h, const float *d, int n, float *buf) {
    REVISAR(cudaMemcpy(buf, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h[i] = buf[i];
}

/* Sube las partículas leídas (double) a la GPU (float). */
static ParticulasGPU subir_particulas(const Particulas *p) {
    ParticulasGPU g; int n = p->n;
    float *buf = (float *)reservar(n * sizeof(float));
    g.n = n;
    g.x = subir_floats(p->x, n, buf);   g.y = subir_floats(p->y, n, buf);   g.z = subir_floats(p->z, n, buf);
    g.vx = subir_floats(p->vx, n, buf); g.vy = subir_floats(p->vy, n, buf); g.vz = subir_floats(p->vz, n, buf);
    g.m = subir_floats(p->m, n, buf);
    REVISAR(cudaMalloc(&g.ax, n * sizeof(float))); REVISAR(cudaMalloc(&g.ay, n * sizeof(float)));
    REVISAR(cudaMalloc(&g.az, n * sizeof(float))); REVISAR(cudaMalloc(&g.tmp, n * sizeof(float)));
    REVISAR(cudaMalloc(&g.id, n * sizeof(int)));  REVISAR(cudaMalloc(&g.tmp_int, n * sizeof(int)));
    REVISAR(cudaMalloc(&g.indice, n * sizeof(int)));
    REVISAR(cudaMalloc(&g.hilo_a_particula, n * sizeof(int)));
    REVISAR(cudaMalloc(&g.codigo, n * sizeof(unsigned long long)));
    REVISAR(cudaMemcpy(g.id, p->id, n * sizeof(int), cudaMemcpyHostToDevice));
    free(buf);
    return g;
}

/* Baja posición, velocidad, masa, aceleración e id para escribir las salidas. */
static void bajar_particulas(Particulas *p, const ParticulasGPU *g) {
    int n = g->n;
    float *buf = (float *)reservar(n * sizeof(float));
    bajar_floats(p->x, g->x, n, buf);   bajar_floats(p->y, g->y, n, buf);   bajar_floats(p->z, g->z, n, buf);
    bajar_floats(p->vx, g->vx, n, buf); bajar_floats(p->vy, g->vy, n, buf); bajar_floats(p->vz, g->vz, n, buf);
    bajar_floats(p->ax, g->ax, n, buf); bajar_floats(p->ay, g->ay, n, buf); bajar_floats(p->az, g->az, n, buf);
    bajar_floats(p->m, g->m, n, buf);
    REVISAR(cudaMemcpy(p->id, g->id, n * sizeof(int), cudaMemcpyDeviceToHost));
    free(buf);
}

/* ============================================================ fases */

static Caja caja_gpu(const ParticulasGPU *g) {
    thrust::device_ptr<float> ejes[3] = {thrust::device_pointer_cast(g->x),
        thrust::device_pointer_cast(g->y), thrust::device_pointer_cast(g->z)};
    double mn[3], mx[3], lado = 0.0;
    for (int e = 0; e < 3; e++) {
        auto par = thrust::minmax_element(ejes[e], ejes[e] + g->n);
        mn[e] = *par.first; mx[e] = *par.second;
        if (mx[e] - mn[e] > lado) lado = mx[e] - mn[e];
    }
    if (lado <= 0.0) lado = 1e-12;
    Caja c = {0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2]), 0.5 * lado};
    return c;
}

/* arr <- arr[indice[k]] usando tmp, e intercambia punteros (no copia de vuelta). */
static void reordenar(float **arr, float **tmp, const int *indice, int n) {
    thrust::device_ptr<const int> ind = thrust::device_pointer_cast(indice);
    thrust::gather(ind, ind + n, thrust::device_pointer_cast(*arr), thrust::device_pointer_cast(*tmp));
    float *aux = *arr; *arr = *tmp; *tmp = aux;
}

/* Caja + códigos + sort_by_key + mover los datos. Devuelve la caja para el árbol. */
static Caja ordenar_gpu(ParticulasGPU *g, int hilos) {
    int bloques = (g->n + hilos - 1) / hilos;
    Caja caja = caja_gpu(g);
    kernel_codigos<<<bloques, hilos>>>(*g, caja);
    REVISAR(cudaGetLastError());
    thrust::device_ptr<unsigned long long> cod = thrust::device_pointer_cast(g->codigo);
    thrust::sort_by_key(cod, cod + g->n, thrust::device_pointer_cast(g->indice));
    float **campos[] = {&g->x, &g->y, &g->z, &g->vx, &g->vy, &g->vz, &g->m};
    for (int c = 0; c < 7; c++) reordenar(campos[c], &g->tmp, g->indice, g->n);
    thrust::device_ptr<const int> ind = thrust::device_pointer_cast((const int *)g->indice);
    thrust::gather(ind, ind + g->n, thrust::device_pointer_cast(g->id), thrust::device_pointer_cast(g->tmp_int));
    int *aux = g->id; g->id = g->tmp_int; g->tmp_int = aux;
    kernel_hilo_a_particula<<<bloques, hilos>>>(*g);
    REVISAR(cudaGetLastError());
    return caja;
}

/* Agranda los arreglos del árbol en la GPU si el de este paso tiene más nodos. */
static void asegurar_capacidad(ArbolGPU *d, int nodos) {
    if (nodos <= d->capacidad) return;
    int *enteros[] = {d->inicio, d->cuenta, d->es_hoja};
    float *reales[] = {d->masa, d->cmx, d->cmy, d->cmz, d->cx, d->cy, d->cz, d->mitad};
    for (int k = 0; k < 3; k++) cudaFree(enteros[k]);
    for (int k = 0; k < 8; k++) cudaFree(reales[k]);
    cudaFree(d->hijos);
    int cap = 2 * nodos;
    REVISAR(cudaMalloc(&d->hijos, 8 * cap * sizeof(int)));
    REVISAR(cudaMalloc(&d->inicio, cap * sizeof(int))); REVISAR(cudaMalloc(&d->cuenta, cap * sizeof(int)));
    REVISAR(cudaMalloc(&d->es_hoja, cap * sizeof(int)));
    float **r[] = {&d->masa, &d->cmx, &d->cmy, &d->cmz, &d->cx, &d->cy, &d->cz, &d->mitad};
    for (int k = 0; k < 8; k++) REVISAR(cudaMalloc(r[k], cap * sizeof(float)));
    d->capacidad = cap;
}

static void subir_reales(float *d, const double *h, int n, float *buf) {
    for (int i = 0; i < n; i++) buf[i] = (float)h[i];
    REVISAR(cudaMemcpy(d, buf, n * sizeof(float), cudaMemcpyHostToDevice));
}

/* Copia el árbol aplanado de la CPU (double) a la GPU (float). */
static void subir_arbol(ArbolGPU *d, const Arbol *a) {
    int n = a->n_nodos;
    asegurar_capacidad(d, n);
    REVISAR(cudaMemcpy(d->hijos, a->hijos, 8 * n * sizeof(int), cudaMemcpyHostToDevice));
    REVISAR(cudaMemcpy(d->inicio, a->inicio, n * sizeof(int), cudaMemcpyHostToDevice));
    REVISAR(cudaMemcpy(d->cuenta, a->cuenta, n * sizeof(int), cudaMemcpyHostToDevice));
    REVISAR(cudaMemcpy(d->es_hoja, a->es_hoja, n * sizeof(int), cudaMemcpyHostToDevice));
    float *buf = (float *)reservar(n * sizeof(float));
    subir_reales(d->masa, a->masa, n, buf);
    subir_reales(d->cmx, a->cmx, n, buf); subir_reales(d->cmy, a->cmy, n, buf); subir_reales(d->cmz, a->cmz, n, buf);
    subir_reales(d->cx, a->cx, n, buf);   subir_reales(d->cy, a->cy, n, buf);   subir_reales(d->cz, a->cz, n, buf);
    subir_reales(d->mitad, a->mitad, n, buf);
    free(buf);
}

/* Árbol en la CPU con c_core/: baja posiciones, masas y códigos ya ordenados, arma el
 * octree y lo sube. Decisión del diseño (preguntada al profesor): construir en la CPU. */
static void construir_en_cpu(Arbol *a, ArbolGPU *d, Particulas *p, const ParticulasGPU *g, Caja caja) {
    int n = g->n;
    float *buf = (float *)reservar(n * sizeof(float));
    bajar_floats(p->x, g->x, n, buf); bajar_floats(p->y, g->y, n, buf);
    bajar_floats(p->z, g->z, n, buf); bajar_floats(p->m, g->m, n, buf);
    REVISAR(cudaMemcpy(p->codigo, g->codigo, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    free(buf);
    construir_arbol(a, p, caja);
    subir_arbol(d, a);
}

/* ============================================================ simulación */

struct Simulacion {
    Particulas p; ParticulasGPU g; Arbol arbol; ArbolGPU d;
    Parametros par; Tiempos t; Cronometro reloj_gpu; int orden;
};

static void calcular_fuerzas(Simulacion *s) {
    int hilos = s->par.workers, bloques = (s->g.n + hilos - 1) / hilos;
    empezar(&s->reloj_gpu);
    Caja caja = ordenar_gpu(&s->g, hilos);
    s->t.ordenamiento += terminar(&s->reloj_gpu);
    empezar(&s->reloj_gpu);
    construir_en_cpu(&s->arbol, &s->d, &s->p, &s->g, caja);
    s->t.construccion += terminar(&s->reloj_gpu);
    empezar(&s->reloj_gpu);
    kernel_recorrido<<<bloques, hilos>>>(s->d, s->g, (float)s->par.theta, (float)s->par.eps, s->orden);
    REVISAR(cudaGetLastError());
    s->t.recorrido += terminar(&s->reloj_gpu);
}

static void integrar(Simulacion *s, int con_drift) {
    int hilos = s->par.workers, bloques = (s->g.n + hilos - 1) / hilos;
    empezar(&s->reloj_gpu);
    kernel_medio_kick<<<bloques, hilos>>>(s->g, (float)s->par.dt);
    if (con_drift) kernel_drift<<<bloques, hilos>>>(s->g, (float)s->par.dt);
    REVISAR(cudaGetLastError());
    s->t.integracion += terminar(&s->reloj_gpu);
}

static int leer_orden_hilos(void) {
    const char *v = getenv("ORDEN_HILOS");
    if (v == NULL) return ORDEN_HILOS_POR_DEFECTO;
    return strcmp(v, "original") == 0 ? ORIGINAL : MORTON;
}

int main(int argc, char **argv) {
    Simulacion s;
    memset(&s, 0, sizeof s);
    s.par = leer_parametros(argc, argv);
    s.orden = leer_orden_hilos();
    REVISAR(cudaEventCreate(&s.reloj_gpu.ini)); REVISAR(cudaEventCreate(&s.reloj_gpu.fin));
    double inicio = reloj();
    s.p = leer_particulas(s.par.entrada);
    s.g = subir_particulas(&s.p);
    s.t.lectura = reloj() - inicio;

    calcular_fuerzas(&s);                       /* aceleraciones del paso 0 */
    bajar_particulas(&s.p, &s.g);
    escribir_aceleraciones(s.par.salida, &s.p);
    for (int k = 0; k < s.par.pasos; k++) {
        integrar(&s, 1);                        /* medio kick + drift */
        calcular_fuerzas(&s);
        integrar(&s, 0);                        /* medio kick con las aceleraciones nuevas */
    }
    REVISAR(cudaDeviceSynchronize());
    s.t.total = reloj() - inicio;               /* comunicacion = 0 (las copias van en construcción) */
    bajar_particulas(&s.p, &s.g);
    escribir_estado_final(s.par.salida, &s.p);
    escribir_tiempos(s.par.salida, &s.t);
    printf("N = %d, pasos = %d, hilos/bloque = %d, orden = %s, theta = %g, nodos = %d, total = %.3f s\n",
           s.p.n, s.par.pasos, s.par.workers, s.orden == MORTON ? "morton" : "original",
           s.par.theta, s.arbol.n_nodos, s.t.total);
    return 0;
}
