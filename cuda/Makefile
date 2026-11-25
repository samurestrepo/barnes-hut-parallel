# Compila la versión CUDA:  make -C cuda   (desde la raíz del repo, en Colab con una T4)
# El núcleo de c_core/ se compila con gcc como C (no como C++) y se enlaza con nvcc.
# -arch=sm_75: arquitectura de la T4 (Turing). -lineinfo: compute-sanitizer muestra la línea.
NVCC   = nvcc
NVFLAGS = -O2 -arch=sm_75 -lineinfo
CC     = gcc
CFLAGS = -O2 -std=c11
NUCLEO = ../c_core/bh_io.c ../c_core/bh_arbol.c ../c_core/bh_fuerzas.c
OBJETOS = bh_io.o bh_arbol.o bh_fuerzas.o

bh_cuda: bh_cuda.cu $(OBJETOS) ../c_core/bh_core.h
	$(NVCC) $(NVFLAGS) -o bh_cuda bh_cuda.cu $(OBJETOS) -lm

%.o: ../c_core/%.c ../c_core/bh_core.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f bh_cuda $(OBJETOS)
