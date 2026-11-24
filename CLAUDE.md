# Proyecto Barnes-Hut — reglas para el agente

## Contexto
Proyecto final de Programación Concurrente y Distribuida. Simulación de N cuerpos
con Barnes-Hut en 4 versiones paralelas. Los humanos deben poder LEER, EXPLICAR y
MODIFICAR este código en pocos minutos, sin IA ni internet, en una demo en vivo.
Eso manda sobre la velocidad y la elegancia.

## Fuente de verdad
- CONTRATO.md: formato de archivos, física, CLI y salidas. No se cambia.
- DISEÑO.md: tareas S0–S7 (Samuel) y C0–C6 (Sebastián). Haz SOLO la tarea pedida.

## Dueños de carpetas
- Samuel: common/, py_core/, py_mp/, mpi/, results/<impl>/, informe/samuel_*.md
- Sebastián: c_core/, c_omp/, cuda/, results/<impl>/, informe/sebastian_*.md
- Trabaja SOLO en la rama de la implementación de la tarea (impl/py-core,
  impl/mp, impl/mpi, impl/c-core, impl/omp, impl/cuda). Si no existe, créala
  desde main. Al terminar la versión, abre un PR a main sin squash con la
  salida del verificador en la descripción.
- Nunca edites carpetas del otro. Si el contrato no alcanza, escribe en
  CONTRATO_DUDAS.md y sigue con una suposición marcada en el código.

## Estilo obligatorio
- Comentarios en español que expliquen el PORQUÉ, no el qué.
- Todas las perillas de la demo (theta, eps, dt, particionado, schedule,
  tamaño de bloque) arriba del archivo o en la CLI, con un comentario.
- Funciones cortas (< 40 líneas), nombres en español claros, nada de trucos.
- Marca con `# PARALELO:` / `// PARALELO:` cada punto donde se reparte trabajo
  y con `# SINCRONIZACIÓN:` / `// SINCRONIZACIÓN:` cada punto donde hay
  comunicación, candado, barrera o posible carrera.
- Mide cada fase y escribe PREFIJO_times.csv como dice el contrato.

## Al terminar una tarea
1. Corre common/verify.py con theta 0 y 0.5 (N = 2000). Si falla, no está hecha.
2. Haz UN commit con el código de la tarea al inicio del mensaje.
3. Escribe en el chat, en 5 líneas, qué hiciste y qué debería poder explicar
   el humano, y hazle 2 preguntas de verificación sobre esa parte.
