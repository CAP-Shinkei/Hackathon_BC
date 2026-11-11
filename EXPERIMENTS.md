# Registro de experimentos (E16 en adelante)

Los experimentos E01–E15 del equipo están en `docs/handoff/EXPERIMENTS.md`. Este fichero continúa
la numeración con el pipeline reconstruido en este repositorio.

## Protocolo

- **Datos**: 10 partidos externos (7 DFL/IDSSE, 3 Metrica sample). Ningún dato de BAR-ATH.
- **Etiquetas**: posesión controlada con regla de 2 s (`mtah/labels.py`).
- **Validación**: dejar-un-partido-fuera. El partido retenido se daña con cada régimen de huecos
  de evaluación y dos semillas (7 y 8).
- **Métrica**: el scorer oficial (`official/evaluation/possession_metrics.py`) a través de
  `mtah/scoring.py`. La cifra de cada experimento es la media por partido.
- **Regla de adopción**: mejora emparejada en ≥ 8 de 10 partidos, ganancia media ≥ 0,01 de
  spellF1, sin retroceso en SkillCorner (huecos reales) ni en macroF1.
- **Reproducir**: `python scripts/run_experiment.py <nombre>`; resultados en `out/results/`.

Regímenes de huecos (`mtah/gapsim.py`):

| Régimen | Qué simula |
|---|---|
| `random35`, `random50` | Huecos cortos al azar; las paradas quedan visibles (supuesto original del equipo) |
| `mixed41`, `mixed41_art` | 41 % de balón ausente, pérdidas por densidad de jugadores y balón rápido; `_art` añade detecciones falsas y jugadores perdidos |
| `dark50` | 50 % ausente, el 70 % de las paradas ocultas (supuesto contrario, parecido a SkillCorner) |

## E16 — Puerta de reconstrucción

Entrenamiento y evaluación con `random35` + `random50`.

| Config | Temperatura | spellF1 | macroF1 | D-F1 |
|---|---|---|---|---|
| e16_t05 | 0,5 | 0,6262 | | |
| e16_base | 1 | 0,6464 | 0,8745 | 0,9269 |
| e16_t2 | 2 | 0,6590 | | |
| e16_t4 | 4 | 0,6654 | | |

Baseline oficial (ventana 50) sobre un partido DFL dañado: ≈ 0,27.

**Lectura**: la reconstrucción funciona. No es comparable directamente con el 0,559 del equipo,
que se midió entre datasets distintos; esto es dentro del mismo conjunto de 10 partidos.
**Decisión**: puerta superada; temperatura 4 como punto de partida del decodificador.

## E17 — Entrenamiento con mezcla de regímenes

Evaluación en el banco `random35` + `mixed41_art` + `dark50`, temperatura 4 salvo indicación.

| Config | Entrena con | spellF1 | macroF1 | D-F1 | random35 | mixed41_art | dark50 |
|---|---|---|---|---|---|---|---|
| e17_old_train | random35, random50 | 0,6067 | 0,8176 | 0,8426 | 0,678 | 0,524 | 0,619 |
| e17_mix_train | random35, mixed41, mixed41_art, dark50 | **0,6743** | **0,8841** | **0,9401** | 0,680 | 0,651 | 0,692 |
| e17_mix_t2 | ídem, temperatura 2 | 0,6678 | 0,8838 | 0,9394 | 0,672 | 0,643 | 0,688 |
| e17_mix_t8 | ídem, temperatura 8 | 0,6621 | 0,8826 | 0,9406 | 0,669 | 0,641 | 0,676 |

Comparación emparejada `e17_mix_train` frente a `e17_old_train`: **+0,068 de media, mejora en
10 de 10 partidos** (DFL +0,07 a +0,10; Metrica +0,02 a +0,04).

**Lectura**: el modelo entrenado solo con huecos al azar se hunde cuando hay artefactos
(0,524) y cuando las paradas están ocultas (0,619). La mezcla lo arregla sin perder nada en el
régimen original (0,678 → 0,680). La temperatura 4 sigue siendo la mejor.
**Decisión**: adoptado `e17_mix_train` como nueva base.
**Aviso**: los regímenes de evaluación coinciden con los de entrenamiento (con otra semilla y
otro partido), así que parte de la ganancia es "conocer el examen". La comprobación independiente
es SkillCorner (huecos reales), más abajo.
