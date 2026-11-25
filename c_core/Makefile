# Compila el Barnes-Hut secuencial:  make -C c_core   (desde la raíz del repo)
# -O2: optimiza sin cambiar resultados; -lm: sqrt y fabs.
CC     = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra
FUENTES = bh.c bh_io.c bh_arbol.c bh_fuerzas.c

bh: $(FUENTES) bh_core.h
	$(CC) $(CFLAGS) -o bh $(FUENTES) -lm

clean:
	rm -f bh
