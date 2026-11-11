# Resumen del trabajo — More than a Hack 2027, Fase 1 (posesión)

Fechas: 10–11 de noviembre de 2026. Plazo de entrega: 17 de noviembre de 2026.
Repositorio: `~/Downloads/mtah-possession` (git local, rama `main`, sin remoto).

## 1. Resultado en una tabla

spellF1 medido con el scorer oficial, siempre en partidos externos (nunca en BAR-ATH).

| | Huecos simulados (10 partidos) | Huecos reales (8 partidos SkillCorner) |
|---|---|---|
| Baseline oficial | ≈ 0,27 | 0,305 |
| Números previos del equipo | 0,559 | 0,454 |
| **Candidato final** | **0,691** | **0,596** |

Salvedades:

- La comparación en simulado no usa el mismo protocolo que el del equipo.
- El 0,596 está inflado: el modelo entrena con otros partidos del mismo proveedor y liga.
- El modelo entregado entrena con los 30 partidos, así que no tiene medida propia sobre datos no
  vistos; las cifras son de la misma configuración con 8 partidos apartados.
- Nada de esto se ha medido en BAR-ATH.

## 2. Punto de partida

- Petición: hacer un plan con lo disponible para mejorar los números del equipo, todo en Python,
  con un enjambre de agentes Haiku 5.5 organizado desde Fable 5.2.
- Lo que había: `mtah-e14-patch/` con 4 scripts (E14/E15, silbato) y 3 documentos. **El pipeline
  original del equipo no estaba**, y el repo oficial del challenge solo trae scorer, baseline y
  cargadores.
- Del dataset solo estaba el XML de metadatos; faltaban tracking, vídeos, homografía y tagging.
- El documento "Pendientes por frente" era una lista personal sin relación con el challenge; se
  ignoró.
- La URL del repo con `20270` no existe; el real es `metrica-sports/more-than-a-hack-2027`.

## 3. Decisiones tomadas

| Decisión | Motivo |
|---|---|
| Los agentes no tocan datos de BAR-ATH; el paso final lo ejecuta una persona | Las reglas prohíben subir datos del partido a LLM alojados. Elegido por el usuario. |
| Reconstruir el pipeline desde cero | El original no estaba disponible. |
| Repo nuevo (`mtah-possession`) en vez de trabajar en `mtah-e14-patch` | El paquete antiguo tenía restos de otras herramientas. |
| scikit-learn `HistGradientBoostingClassifier` en vez de LightGBM | LightGBM no cargaba en este equipo (falta `libomp`). Desviación del plan. |
| Usar el scorer oficial como único scorer | Que una cifra externa signifique lo mismo que en el leaderboard. |
| Mantener los 3 partidos de Metrica sample-data | Son otros partidos y de otro repo; confirmado por el usuario. |

## 4. Qué se construyó

Biblioteca `mtah/`:

- `schema.py` — formato común de partido en memoria (metros, 25 fps, NaN = ausente).
- `external/dfl.py`, `external/metrica.py`, `external/skillcorner.py` — descarga y carga de 7 + 3
  + 20 partidos públicos, con etiquetas.
- `gapsim.py` — simulación de huecos: `random35`, `random50`, `crowd41`, `mixed41`,
  `mixed41_art`, `dark50`.
- `features.py` (89 columnas) y `features_v2.py` (152 columnas) — pistas por frame, simétricas
  entre equipos.
- `model.py` — clasificador por frame con aumento por intercambio de equipos.
- `decode.py` — decodificador semi-Markov de duración explícita y ajuste de fronteras.
- `labels.py` (posesión controlada de 2 s), `rules.py`, `ballfill.py`, `ballclean.py`.
- `io_atd.py`, `atd_export.py` — lectura y escritura del formato de tracking de la competición.
- `submission.py` — entrenamiento, guardado y predicción del modelo; escritura del CSV.
- `experiment.py`, `scoring.py`, `baseline.py` — protocolo dejar-un-partido-fuera, scorer y
  baseline oficiales.

Scripts: `train.py`, `make_submission.py`, `run_experiment.py`, `validate_real_gaps.py`.

Documentos: `README.md`, `TECHNICAL.md` (borrador), `EXPERIMENTS.md` (E16–E24).

Modelo: `model/bundle.joblib` (6,7 MB), versionado en el repo.

## 5. Experimentos

Regla de adopción fijada antes de medir: mejora en ≥ 8 de 10 partidos, ganancia media ≥ 0,01,
sin retroceso en SkillCorner ni en macroF1.

| Exp. | Qué probó | Simulado | Real | Decisión |
|---|---|---|---|---|
| E16 | Reconstrucción del pipeline (puerta) | 0,646 → 0,665 con temperatura 4 | — | Puerta superada |
| E17 | Entrenar con mezcla de regímenes de huecos | 0,607 → 0,674 (10/10) | 0,437 → 0,468 | **Adoptado** |
| E18 | Pistas ampliadas | 0,674 → 0,697 (10/10) | 0,468 → 0,482 | **Adoptado** |
| E19 | Ajuste del decodificador (5 variantes) | ≤ +0,002 | — | No adoptado |
| E21 | Media de 5 semillas | +0,001 (5/10) | — | No adoptado |
| E24 | Entrenar también con SkillCorner | 0,697 → 0,691 (2/10) | 0,482 → 0,596 | **Adoptado como excepción** |
| E22 | Filtro de balón físicamente imposible | +0,002 (8/10) | +0,004 | No adoptado |
| E20 | Regla de posesión 0/1/2/3 s | — | — | No realizado |
| E23 | Red temporal | — | — | No realizado |

Sobre la excepción de E24: la regla pedía cero retroceso en simulado y hay −0,006. Se adoptó
porque no se sabe a cuál de los dos supuestos se parece BAR-ATH, el único dato publicado (balón
ausente en el 41 % de los frames) coincide con SkillCorner (≈ 40 %), y lo que se arriesga es unas
veinte veces menor que lo que se gana. Para aplicar la regla al pie de la letra:
`scripts/train.py --real-games 0`.

Hallazgo que orientó E17 y E24: en SkillCorner el balón falta en ≈ 80 % de los frames de balón
muerto frente a ≈ 6 % de los de balón vivo, con huecos largos. Es lo contrario de lo que asumía
el simulador original (huecos cortos al azar, paradas visibles).

## 6. Revisión de código

Enjambre de 14 agentes Haiku: 3 revisores con focos distintos, un verificador por hallazgo, uno
de pruebas y uno de documentación. 8 hallazgos confirmados.

Arreglados:

- El decodificador aceptaba probabilidades NaN y etiquetaba el bloque entero como local.
- El lector de tracking descartaba líneas truncadas sin avisar.
- `make_submission.py` sobrescribía el CSV anterior antes de validar.
- `train.py` no comprobaba el rango de `--real-games`.
- Las claves de caché no incluían parámetros del régimen, etiquetas ni conjunto de entrenamiento.
- `train.py` fallaba con las pistas ampliadas (detectado al ejecutarlo entero por primera vez).

Anotado como límite, sin cambio: `dark50` y `mixed41*` ocultan paradas usando las etiquetas
reales, así que sus cifras simuladas son optimistas.

## 7. Verificación

- Suite de pruebas completa en verde.
- Cobertura de `mtah/`: 95 % según el agente de pruebas, medida antes de los últimos arreglos y
  no repetida después. `mtah/baseline.py` sin cobertura propia.
- Entrenamiento final ejecutado dos veces: mismo md5
  (`6565e767897db5160486f878e0b6ab1c`).
- `make_submission.py` con el modelo final sobre un partido externo exportado al formato de la
  competición: 6 s, pasa la validación oficial. Prueba de mecánica, no de calidad.

## 8. Cumplimiento y cosas a declarar

- Ningún agente ha leído tracking, vídeo, tagging ni ground truth de BAR-ATH. La carpeta del
  dataset está bloqueada en `.claude/settings.json`.
- **Excepción**: antes de fijar esa barrera, un agente leyó el XML de metadatos
  `tactical/BAR-ATH_tactical_atd.xml` (lista de pistas y equipos, sin posiciones ni etiquetas).
- Del repo oficial se usa solo código (scorer, baseline, `games.py`). No se usan su ground truth
  público de 5 minutos ni sus CSV de baseline.
- Queda por aclarar, por parte del equipo, el uso histórico de los 5 minutos públicos y qué datos
  se pasaron a herramientas de IA en el trabajo anterior (E01–E15).

## 9. Pendiente

De una persona del equipo:

1. Descargar las partes 001–005 del dataset fuera del repo.
2. Ejecutar:
   ```bash
   .venv/bin/python scripts/make_submission.py --data-dir <carpeta del dataset> --out <equipo>.csv
   .venv/bin/python official/evaluation/validate_submission.py <equipo>.csv
   ```
3. Revisar los recuentos del validador: se esperan unas 250–350 posesiones. Muchas menos, o casi
   todo `D`, indica un problema con el tracking real.
4. Cerrar los puntos **[CONFIRMAR]** de `TECHNICAL.md` (licencias de Metrica y SkillCorner,
   silbato, 5 minutos públicos).
5. Publicar el repo comprobando que no lleva datos del partido.
6. Subir CSV, URL y documento al formulario, nombrados con el nombre del equipo.

Opcional:

- E20 (sensibilidad a la regla de 2 s).
- Volver a medir la cobertura.
- Decidir el frame del saque del segundo tiempo: 113741 (`games.py`, el usado) o 113742 (FAQ).

Sin resolver: en la conversación apareció un "54" sin contexto. Si es el número de posesiones
sobre BAR-ATH, es una señal de alarma y hay que diagnosticarlo antes de entregar.

## 10. Historial de commits

```
fc7e248 feat: pipeline modules, external loaders and leave-one-game-out protocol
6b5286e feat: SkillCorner loader, features v2, ball plausibility filter, submission scripts with end-to-end test
14b31bf feat: feature-version switch, real-gap validation script, experiment log E16-E17
972859c feat: decoder and seed configs, real-gap training option, experiment log E18-E19
15002db feat: train with real-gap games, ball-cleaning option, experiment log E21 and E24
db7638b fix: reject non-finite probabilities, warn on skipped tracking lines, validate before replacing the submission; add coverage tests and README
6a2ae73 docs: technical document draft
71a2fae fix: cache keys cover regime parameters, labels and training pool; log E22
326a2e7 fix: bundle training works with the extended feature module
ad63e00 feat: trained model bundle (30 external games, deterministic)
```
