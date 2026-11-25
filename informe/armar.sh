#!/usr/bin/env bash
# Arma informe/informe.pdf a partir de las secciones .md (desde la raíz del repo: bash informe/armar.sh).
# Requiere pandoc y xelatex (Ubuntu: sudo apt install pandoc texlive-xetex texlive-lang-spanish).
# El enunciado pide MÁXIMO 6 PÁGINAS: el script imprime cuántas salieron.
set -euo pipefail
cd "$(dirname "$0")"

# Orden del informe: organización, las 4 versiones, verificación, mediciones comunes.
SECCIONES=(
  00_intro.md
  samuel_mp.md
  samuel_mpi.md
  sebastian_omp.md
  sebastian_cuda.md
  05_verificacion.md
  06_mediciones.md
)

pandoc "${SECCIONES[@]}" \
  --pdf-engine=xelatex \
  --resource-path=..:. \
  -V geometry:margin=2cm \
  -V fontsize=11pt \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -o informe.pdf

if command -v pdfinfo >/dev/null; then
  echo "Páginas: $(pdfinfo informe.pdf | awk '/^Pages/ {print $2}') (máximo 6)"
fi
