# mtah-possession

Clasifica cada fotograma de un vídeo táctico de fútbol como posesión local, posesión visitante o balón parado (`H`, `A`, `D`). El pipeline entrena un clasificador con partidos externos dañados de forma simulada, decodifica la posesión con priores temporales y escribe el CSV de envío (`frame,label`).

## Requisitos e instalación

- Python 3.12.
- Clonar el repositorio oficial de la competición dentro de `official/` (lo usan `scripts/make_submission.py`, `mtah/scoring.py` y los tests):

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 official
```

- Crear el entorno virtual e instalar las dependencias:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` está generado a partir de `requirements.in` con uv. Si se cambian dependencias, editar `requirements.in` y regenerar.

## Descargar datos externos

Los datos externos de entrenamiento y validación se guardan en `external_data/` (ignorado por git). No tienen punto de entrada en línea de comandos: cada loader expone una función `download()` que se llama desde Python. Desde la raíz del repositorio:

```bash
.venv/bin/python -c "from mtah.external import dfl, metrica, skillcorner; dfl.download(); metrica.download(); skillcorner.download()"
```

- `dfl.download()` descarga los XML de 7 partidos desde Figshare a `external_data/dfl/`.
- `metrica.download()` descarga los ficheros de 3 partidos de muestra de Metrica a `external_data/metrica/`.
- `skillcorner.download()` descarga 20 partidos de SkillCorner a `external_data/skillcorner/`. El tracking viene en git-lfs; la función comprueba que no se haya guardado un puntero LFS en lugar del fichero real.

Los ficheros ya presentes se conservan. `train.py` requiere estos tres descargas.

## Datos de la competición

Los datos del partido de la competición no forman parte del repositorio y deben mantenerse fuera de él. `make_submission.py` los lee desde una carpeta que se pasa con `--data-dir` o con la variable de entorno `MTAH_DATA_DIR`. Ni esa carpeta ni su contenido deben copiarse dentro del repositorio ni subirse a git.

## Entrenamiento

```bash
.venv/bin/python scripts/train.py
```

Todos los flags son opcionales. Valores por defecto (de su `argparse`):

| Flag | Por defecto | Significado |
|------|-------------|-------------|
| `--regimes` | `random35 mixed41 mixed41_art dark50` | regímenes de daño simulado para entrenar |
| `--model-seeds` | `0` | semillas del modelo |
| `--temperature` | `4.0` | temperatura de la decodificación |
| `--step` | `5` | paso de muestreo de fotogramas |
| `--min-frames` | `0` | duración mínima de una posesión |
| `--features` | `features_v2` | módulo de featurización en `mtah/` |
| `--real-games` | `20` | partidos de SkillCorner (reales) que se añaden, tomados del final de la lista |
| `--out` | `model/bundle.joblib` | dónde se guarda el bundle |

Notas:

- Los partidos de entrenamiento controlado se cargan igual que en `run_experiment.py`, con la regla de posesión de 2 s.
- El valor por defecto de `--real-games` usa los 20 partidos de SkillCorner, incluidos los 8 primeros que `validate_real_gaps.py` reserva como validación. El bundle resultante sirve para generar el envío, pero no es una medida limpia sobre esos 8 partidos.

## Generar el envío

```bash
.venv/bin/python scripts/make_submission.py --data-dir /ruta/a/los/datos/de/la/competicion
```

Flags (de su `argparse`):

| Flag | Por defecto | Significado |
|------|-------------|-------------|
| `--data-dir` | `$MTAH_DATA_DIR` | carpeta con los datos de la competición (obligatorio si no hay variable de entorno) |
| `--bundle` | `model/bundle.joblib` | bundle entrenado |
| `--out` | `out/submissions/submission.csv` | CSV de salida |
| `--game` | `BAR-ATH` | id del partido en `official/games.py` |

El script predice, escribe el CSV y lo valida con `pm.validate_submission`. Si la validación falla, aborta con error. Al final imprime la ruta, el md5, el número de filas, el recuento de etiquetas `H A D X` y el número de posesiones. No imprime datos a nivel de fotograma.

Validación oficial: ejecutar `official/evaluation/validate_submission.py` sobre el CSV generado. Los argumentos de ese script no se documentan aquí; consultar su cabecera en el repositorio oficial. Este paso lo ejecuta una persona.

## Experimentos y tests

Experimentos (protocolo dejar-un-partido-fuera). Son lentos; no se lanzan desde un agente de IA:

```bash
.venv/bin/python scripts/run_experiment.py e16_base          # escribe out/results/e16_base.json
.venv/bin/python scripts/validate_real_gaps.py e17_mix_train  # escribe out/results/e17_mix_train_real.json
```

Los nombres válidos son las claves de `CONFIGS` en `scripts/run_experiment.py`. El registro de resultados está en `EXPERIMENTS.md` (E16 en adelante). Los experimentos E01–E15 están en `docs/handoff/EXPERIMENTS.md`.

Tests:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=mtah
```

`pytest.ini` define `testpaths = tests` y `-q`. Los tests de DFL y Metrica con datos reales se omiten si no están descargados. `tests/test_e2e.py` necesita `official/`.

## Estructura del repositorio

| Ruta | Contenido |
|------|-----------|
| `mtah/` | biblioteca: esquema, simulación de huecos, featurización, modelo, decodificación y submission |
| `mtah/schema.py` | formato de datos común (`Match`, códigos `H`, `A`, `D`) |
| `mtah/gapsim.py` | simulación de pérdidas de tracking por régimen de daño |
| `mtah/features.py`, `mtah/features_v2.py` | featurización por fotograma (v2 amplía v1) |
| `mtah/model.py` | clasificador por fotograma, simétrico en los dos equipos |
| `mtah/decode.py` | decodificación semi-Markov por bloques |
| `mtah/labels.py` | etiquetas de posesión controlada (regla de 2 s) |
| `mtah/ballclean.py`, `mtah/ballfill.py` | limpieza y relleno del balón |
| `mtah/rules.py` | reglas geométricas previas al scoring |
| `mtah/baseline.py` | baseline oficial, para comparar |
| `mtah/scoring.py` | llamada fina al scorer oficial |
| `mtah/submission.py` | entrenamiento, guardado y predicción del bundle; escritura del CSV |
| `mtah/experiment.py` | protocolo dejar-un-partido-fuera |
| `mtah/io_atd.py`, `mtah/atd_export.py` | lectura y escritura del formato ATD de la competición |
| `mtah/external/` | loaders de DFL (`dfl.py`), Metrica (`metrica.py`) y SkillCorner (`skillcorner.py`) |
| `scripts/train.py` | entrenamiento del bundle |
| `scripts/make_submission.py` | generación y validación del CSV de envío |
| `scripts/run_experiment.py` | ejecuta configuraciones de experimento con nombre |
| `scripts/validate_real_gaps.py` | entrena con datos simulados y evalúa sobre SkillCorner |
| `tests/` | suite pytest |
| `EXPERIMENTS.md` | registro de experimentos E16 en adelante |
| `docs/handoff/`, `docs/legacy/` | experimentos previos (E01–E15) y material antiguo |
| `experiments/` | carpeta de experimentos (no documentada en este README) |
| `requirements.in`, `requirements.txt`, `pytest.ini` | dependencias y configuración de pytest |
| `official/` | clon del repositorio oficial (ignorado por git) |
| `external_data/` | datos externos descargados (ignorado por git) |
| `out/` | resultados y envíos generados (ignorado por git) |
| `model/` | bundle entrenado, `model/bundle.joblib` por defecto |

## Reglas de trabajo

- Los datos de la competición quedan fuera del repositorio.
- El paso final (validación oficial y envío) lo ejecuta una persona, nunca un agente de IA.
- No hacer commit de `out/`, `external_data/`, `official/` ni `data/`.
