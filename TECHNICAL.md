# Documento técnico — More than a Hack 2027, Fase 1 (posesión)

> Borrador para revisión del equipo. Los puntos marcados **[CONFIRMAR]** necesitan que una
> persona del equipo los verifique antes de entregar.

## 1. Enfoque

Etiquetamos cada frame como posesión local (`H`), visitante (`A`) o balón muerto (`D`) en tres
pasos:

1. **Pistas por frame** a partir solo del tracking (jugadores y balón): distancias jugador–balón
   por equipo, velocidad y posición del balón, forma y avance de los bloques de cada equipo,
   tiempo desde y hasta el último avistamiento del balón, quietud de los jugadores y contexto
   temporal pasado y futuro. Las pistas son simétricas: intercambiar los equipos intercambia las
   columnas.
2. **Clasificador por frame** (`HistGradientBoostingClassifier` de scikit-learn) que da
   P(H), P(A), P(D). Se entrena con cada partido visto desde los dos lados y se predice
   promediando ambas vistas, de modo que no puede preferir a un equipo por ser "local".
3. **Decodificador semi-Markov de duración explícita** (Viterbi) por cada mitad, con
   distribuciones de duración y transiciones contadas en los partidos de entrenamiento, y un
   ajuste final de cada frontera a resolución de frame.

**Por qué así.** La métrica (`spellF1`) premia recuperar posesiones enteras: un frame mal
etiquetado en medio parte una posesión en dos, y una frontera 1 s tarde baja mucho el IoU. Un
clasificador por frame solo no basta (el baseline oficial lo demuestra: mismo algoritmo, la
métrica cambia por un factor de once según el suavizado). El decodificador impone duraciones
plausibles y el clasificador aporta la evidencia.

**El problema central son los huecos.** El balón no aparece en el 41 % de los frames evaluados.
Los partidos públicos con etiquetas tienen el balón siempre visible, así que entrenamos
dañándolos con huecos simulados bajo varios supuestos a la vez (huecos cortos al azar; pérdidas
en zonas pobladas y balones rápidos; detecciones falsas; paradas ocultas) y añadimos partidos con
huecos reales de retransmisión.

## 2. Activos usados y descartados

| Activo proporcionado | Uso |
|---|---|
| Tracking táctico (ATD) | **Única entrada del modelo.** |
| Constantes de `games.py` (bloques evaluados, saque del segundo tiempo) | Los frames fuera de las dos mitades, y los anteriores al saque del segundo tiempo, se etiquetan `D`. |
| Tracking de retransmisión | No usado. |
| Vídeos (táctico y retransmisión) | No usados. |
| Homografía | No usada. |
| Smart Tagging | No usado en la entrega. |
| Audio (silbato) | Probado por el equipo en E14–E15 y descartado: un pitido no distingue parada de reanudación y no mejoró el resultado. **[CONFIRMAR]** |
| Ground truth público de 5 minutos | No usado para entrenar ni para elegir nada. Solo comprobación de cordura del fichero final. **[CONFIRMAR, ver sección 7]** |

Descartado tras medirlo en partidos externos (detalle en `EXPERIMENTS.md` y
`docs/handoff/EXPERIMENTS.md`): cortes duros en los huecos, silbato como parada o como frontera,
relleno aprendido del balón, media de varias semillas (E21, +0,001), fusión de posesiones cortas
y otros ajustes del decodificador (E19, ≤ +0,002).

## 3. Datos de entrenamiento

Ningún dato etiquetado procede del partido proporcionado. Todo son otros partidos, de fuentes
públicas:

| Fuente | Partidos | Etiquetas: quién y cómo | Licencia |
|---|---|---|---|
| DFL / IDSSE (Bassek et al., 2025), figshare | 7 (Bundesliga 1 y 2) | Del proveedor: posesión y estado del balón por frame en el propio tracking | CC BY 4.0 |
| Metrica Sports sample-data (repositorio público) | 3 (anónimos) | Derivadas por nosotros de los eventos del proveedor (pases, recuperaciones, balones fuera…) | Ver repositorio **[CONFIRMAR licencia]** |
| SkillCorner opendata | 20 (A-League 2024/25), tracking de retransmisión a 10 fps remuestreado a 25 | Derivadas por nosotros de las fases de juego del proveedor; los tramos entre fases son `D`; lo desconocido no se evalúa | Ver repositorio **[CONFIRMAR licencia]** |

No hay etiquetas manuales nuestras en ningún partido.

Uso: los 10 partidos de DFL y Metrica se dañan con cuatro regímenes de huecos
(`random35`, `mixed41`, `mixed41_art`, `dark50`, en `mtah/gapsim.py`); los de SkillCorner se usan
tal como vienen, porque sus huecos ya son reales.

## 4. Reglas de posesión

- **Posesión controlada (2 s).** Una posesión de menos de 2 s se asigna al equipo anterior (o al
  siguiente, si viene de una parada). Así un despeje, un rebote o un balón dividido no crean una
  posesión nueva: la posesión cambia cuando el otro equipo la controla.
- **Balón suelto y disputas**: siguen perteneciendo al último equipo que controló, hasta que el
  rival controle o el balón muera.
- **Balón muerto (`D`)**: desde que el juego se para hasta que se reanuda. La reanudación (saque
  de banda, falta, córner, saque de puerta) abre posesión del equipo que saca.
- **Fuera de las mitades** y antes del saque del segundo tiempo: `D`.
- Estas reglas se aplican a las etiquetas de entrenamiento; el modelo las aprende, no hay reglas
  escritas a mano sobre el partido proporcionado ni constantes atadas a frames vistos en vídeo.

## 5. Resultados en partidos externos

Medidos con el scorer oficial. Protocolo: dejar-un-partido-fuera sobre 10 partidos (DFL +
Metrica) con huecos simulados, y 8 partidos de SkillCorner nunca usados en entrenamiento para
huecos reales.

| Modelo | Simulado, spellF1 | Simulado, macroF1 | Real, spellF1 | Real, macroF1 |
|---|---|---|---|---|
| Baseline oficial (ventana 50) | ≈ 0,27 | | 0,305 | |
| Entrenado con huecos al azar | 0,607 | 0,818 | 0,437 | 0,817 |
| + mezcla de regímenes (E17) | 0,674 | 0,884 | 0,468 | 0,841 |
| + pistas ampliadas (E18) | 0,697 | 0,892 | 0,482 | 0,846 |
| + partidos con huecos reales (E24, **entregado**) | 0,691 | 0,890 | 0,596 | 0,886 |

El modelo entregado se entrena con los 30 partidos (incluidos los 8 de validación), así que no
tiene una medida propia sobre datos no vistos; las cifras de la tabla son las de la misma
configuración con esos 8 partidos apartados.

## 6. Limitaciones

- **No sabemos cómo son los huecos del partido evaluado.** Ningún agente de IA ha mirado su
  tracking (sección 7), así que el modelo no está calibrado sobre él. Entrenamos para aguantar
  varios supuestos, pero si los huecos reales siguen otro patrón el resultado bajará.
- **La cifra con huecos reales está inflada**: los partidos de validación de SkillCorner son del
  mismo proveedor y liga que los de entrenamiento.
- **Caída de simulado a real** (0,69 → 0,60): SkillCorner ve solo 4–5 jugadores por equipo; parte
  de la caída no debería darse con cámara táctica, pero no lo hemos podido medir.
- **Definición de posesión**: la regla de 2 s es nuestra lectura; si el ground truth privado usa
  otro umbral, habrá posesiones partidas o fundidas de forma sistemática. No medimos la
  sensibilidad a 0/1/3 s.
- **Cambios de identidad y jugadores perdidos**: el modelo no usa identidades, así que le afectan
  poco, pero tampoco aprovecha quién es el portero o quién saca.
- **Con más tiempo**: un modelo temporal (red recurrente o convolucional) en lugar de pistas con
  ventanas fijas; usar el vídeo para confirmar reanudaciones; calibrar la simulación de huecos
  con estadísticas agregadas del tracking proporcionado.

## 7. Uso de IA y cumplimiento

- El código de este repositorio se escribió con asistencia de un LLM alojado (Claude). Para
  respetar la prohibición de subir datos de la competición a LLM alojados, **los agentes
  trabajaron solo con código y con partidos externos públicos**; la carpeta del dataset estaba
  bloqueada por configuración (`.claude/settings.json`) y el paso que lee el partido
  (`scripts/make_submission.py`) lo ejecutó una persona en local.
- **Excepción conocida**: antes de fijar esa barrera, un agente leyó el fichero de metadatos
  `tactical/BAR-ATH_tactical_atd.xml` (lista de pistas y equipos; no contiene posiciones ni
  etiquetas). No se usó para ninguna decisión del modelo.
- **[CONFIRMAR por el equipo]**: en el trabajo anterior a este repositorio (E01–E15) queda por
  aclarar el uso que se hizo de los 5 minutos públicos y qué datos se pasaron a herramientas de
  IA. Este repositorio no hereda modelos, umbrales ni decisiones de ese trabajo salvo la lista de
  ideas descartadas. Si hay dudas, consultar a la organización antes de entregar.

## 8. Reproducción

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 official
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from mtah.external import dfl, metrica, skillcorner; dfl.download(); metrica.download(); skillcorner.download()"
.venv/bin/python scripts/train.py                      # escribe model/bundle.joblib
.venv/bin/python scripts/make_submission.py --data-dir <carpeta del dataset> --out <equipo>.csv
.venv/bin/python official/evaluation/validate_submission.py <equipo>.csv
```

El modelo entrenado (`model/bundle.joblib`) está versionado en el repositorio, de modo que los
dos últimos comandos bastan para regenerar el mismo fichero. Detalle en `README.md`.
