# Parche E14: señales de balón parado del trabajo 31-oct, revisadas bajo las reglas del 3-nov

Aplicar sobre el repo de Manuel copiando estos archivos encima (paths idénticos).

Nuevos: mtah/whistle.py, experiments/e14_whistle_detection.py, experiments/e14_gap_precision.py.
Modificados: EXPERIMENTS.md (entrada E14), README.md (nota E14).
NO se modifica make_submission.py: el candidato E13 queda intacto por construcción (este parche
no toca el pipeline de submission; no hace falta ni se afirma ninguna identidad byte a byte de
la integración, porque la integración ya no existe).

## Qué sobrevive del trabajo del 31-oct y qué no

- Cortes por huecos de balón (tagboundaries 2 s / 20 s): MEDIDOS y RECHAZADOS. Con el simulador
  de huecos sobre la muestra externa de Metrica, P(muerto | hueco >= 2 s) = 0.51-0.66 frente a
  una tasa base de 0.41; ni siquiera 20 s es limpio (0.81-0.97). El condicionado por Smart
  Tagging no es validable en externos (ningún partido externo lo tiene). Módulo eliminado; la
  tabla y el razonamiento están en EXPERIMENTS.md E14, reproducibles con
  experiments/e14_gap_precision.py (MTAH_LABELS y METRICA_SAMPLE_DIR apuntando a la muestra
  externa de Metrica).
- Detector de silbato: se mantiene SOLO como exportador de eventos experimental, en local, sin
  integración en la submission. Motivo de quitar la integración (forzar D): el silbato detiene
  el juego PERO TAMBIÉN autoriza reanudaciones (saques iniciales, libres pitados; IFAB Ley 9 en
  ambos sentidos), así que un intervalo acústico detectado no demuestra balón muerto. Hacer esa
  distinción exigiría evaluación a nivel de posesión sobre partidos externos CON audio, que no
  existe en los datasets disponibles, y comprobarla contra el vídeo o el output del partido está
  prohibido por las reglas clarificadas.

## Qué SÍ está verificado (y cómo reproducirlo)

- Detección acústica en audio EXTERNO (Wikimedia Commons, dos ficheros citados en el script):
  experiments/e14_whistle_detection.py recupera 5/5 pitidos a +10/+5/0/-5 dB mezclados en grada
  y da 0 falsas alarmas en 227 s de grada pura. Constantes fijas: banda 2100-6400 Hz, umbral
  0.5, tono sostenido >= 0.256 s. El cuantil adaptativo por grabación del prototipo del 31-oct
  se eliminó (habría sido ajustar sobre el audio del partido).
- La medición de precisión de cortes por huecos (tabla E14) se reproduce con
  experiments/e14_gap_precision.py tal cual se entrega.
- Nada de este parche sube datos del partido a ningún servicio: el audio se extrae y procesa en
  local (ffmpeg + numpy), y los scripts imprimen solo agregados.

## E15: ¿ayuda el silbato al candidato? Medido (simulación externa) - NO

Nuevo pedido de Álvaro (9-nov): probar si meter el silbato ayuda. experiments/e15_whistle_benefit.py
lo simula de forma lícita: los eventos de los 3 partidos públicos de Metrica marcan los parones
pitados (31/33/27 por partido); se divide la decodificación en los pitidos simulados (nada
forzado a D), con variantes que degradan la señal (jitter de ±1 s, recall 0.85, falsas alarmas
uniformes y estructuradas cerca de tiros). Resultado, spellF1 medio sobre los 3 partidos:
35% de huecos -> base 0.603 vs 0.601-0.603 con silbato; 50% -> base 0.586 vs 0.586-0.587.
Ni la señal ORÁCULO (pitido perfecto) mueve la métrica más de ±0.003: el decoder HSMM ya coloca
bien las fronteras de parada (F1 de balón muerto 0.88-0.91). Es una simulación de la señal, no
una prueba del detector real (tiempos acústicos y falsas alarmas reales sin medir), pero invierte
la carga de la prueba: no hay evidencia externa que justifique integrar el silbato en el candidato.

## Qué NO está verificado (limitaciones, no las cubre este parche)

- Tasa de falsas alarmas en un partido real (pitos de grada, tambores): no medible sin procesar
  el audio del partido, y aun midiéndola no habría evaluación externa de posesión que justifique
  usarla para cortar.
- Alineación temporal: los tiempos del CSV asumen audio alineado a t=0 del vídeo (extracción
  ffmpeg directa, 25 fps). Verificar ese offset contra el vídeo del partido sería calibración
  sobre el target y NO está permitido; la alineación queda sin verificar.
- Utilidad para la métrica: no hay evidencia de que estos eventos mejoren spellF1; no adoptar
  ningún uso del CSV por inspección del vídeo o del output del partido.

## Nota de procedencia (mensajes reenviados por Álvaro, 9-nov, aparentemente de Manuel)

El entrenamiento usa gap augmentation inspirada en inpainting de imágenes (huecos realistas en
datos limpios), interpolación lineal dentro del hueco (un estimador entrenado no mejoró y se
descartó), selección de versiones solo en partidos públicos Bundesliga/Metrica, mejora reportada
0.24 -> 0.56 mantenida en otros públicos, y el partido del concurso tratado solo como test.
Es coherente con el repo; sigue abierta la duda de Smart Tagging (sin externos que lo tengan).
