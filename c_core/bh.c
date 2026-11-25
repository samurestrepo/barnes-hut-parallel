/* bh.c — Barnes-Hut SECUENCIAL en C (tarea C1). Es la base de c_omp/ y cuda/.
 *
 *   make -C c_core
 *   ./c_core/bh --input data/uniform_N2000_s42.txt --steps 10 --theta 0.5 --out /tmp/c
 *   python common/verify.py correr --cmd "./c_core/bh"
 *
 * Perillas: θ, ε y dt por CLI (ver CONTRATO.md). HOJA_MAX y BITS_EJE en bh_core.h.
 * --directa usa la suma O(N²) en vez del árbol: sirve para depurar. */
#include <stdio.h>
#include "bh_core.h"

/* Las 4 fases que preceden a cada cálculo de aceleraciones, cada una con su reloj.
 * El orden importa: la caja define los códigos, los códigos el orden, y el árbol
 * se arma sobre las partículas ya ordenadas. */
static void calcular_fuerzas(Particulas *p, Arbol *arbol, const Parametros *par, Tiempos *t) {
    double t0 = reloj();
    Caja caja = calcular_caja(p);
    calcular_codigos(p, caja);
    ordenar_por_morton(p);
    double t1 = reloj();
    construir_arbol(arbol, p, caja);
    double t2 = reloj();
    if (par->directa) suma_directa(p, par->eps);
    else              calcular_aceleraciones(arbol, p, par->theta, par->eps);
    double t3 = reloj();
    t->ordenamiento += t1 - t0;
    t->construccion += t2 - t1;
    t->recorrido    += t3 - t2;
}

/* Un paso de leapfrog kick-drift-kick. */
static void paso(Particulas *p, Arbol *arbol, const Parametros *par, Tiempos *t) {
    double t0 = reloj();
    medio_kick(p, par->dt);        /* con las aceleraciones del paso anterior */
    drift(p, par->dt);
    t->integracion += reloj() - t0;
    calcular_fuerzas(p, arbol, par, t);
    t0 = reloj();
    medio_kick(p, par->dt);        /* con las aceleraciones nuevas */
    t->integracion += reloj() - t0;
}

int main(int argc, char **argv) {
    Parametros par = leer_parametros(argc, argv);
    Tiempos t = {0};
    Arbol arbol = {0};   /* vacío: crecer_arbol reserva en el primer uso */
    double inicio = reloj();

    Particulas p = leer_particulas(par.entrada);
    t.lectura = reloj() - inicio;

    calcular_fuerzas(&p, &arbol, &par, &t);   /* aceleraciones del paso 0 */
    escribir_aceleraciones(par.salida, &p);
    for (int s = 0; s < par.pasos; s++)
        paso(&p, &arbol, &par, &t);

    t.total = reloj() - inicio;   /* comunicacion = 0: un solo proceso */
    escribir_estado_final(par.salida, &p);
    escribir_tiempos(par.salida, &t);
    printf("N = %d, pasos = %d, theta = %g, nodos = %d, total = %.3f s\n",
           p.n, par.pasos, par.theta, arbol.n_nodos, t.total);
    liberar_arbol(&arbol);
    liberar_particulas(&p);
    return 0;
}
