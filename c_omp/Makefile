# Compila la versión OpenMP:  make -C c_omp   (desde la raíz del repo)
# Reutiliza el núcleo de c_core/ salvo su main (bh.c). -fopenmp activa los #pragma omp.
CC     = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -fopenmp
NUCLEO = ../c_core/bh_io.c ../c_core/bh_arbol.c ../c_core/bh_fuerzas.c

bh_omp: bh_omp.c $(NUCLEO) ../c_core/bh_core.h
	$(CC) $(CFLAGS) -o bh_omp bh_omp.c $(NUCLEO) -lm

clean:
	rm -f bh_omp
