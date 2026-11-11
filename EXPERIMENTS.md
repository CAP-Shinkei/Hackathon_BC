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

## E19 — Ajuste del decodificador

Mismas probabilidades que `e17_mix_train` (0,6743); solo cambia el decodificador.

| Config | Cambio | spellF1 | macroF1 |
|---|---|---|---|
| e19_t3 | temperatura 3 | 0,6732 | 0,8841 |
| e19_t6 | temperatura 6 | 0,6679 | 0,8833 |
| e19_min25 | posesiones < 1 s se funden con la anterior | 0,6744 | 0,8841 |
| e19_min50 | posesiones < 2 s se funden con la anterior | 0,6760 | 0,8839 |
| e19_step3 | paso de 3 frames en vez de 5 | 0,6738 | 0,8840 |

**Lectura**: el decodificador está en una meseta. Ninguna variante llega al umbral de +0,01; la
mejor (+0,002) está dentro del ruido.
**Decisión**: no se adopta nada. Se mantienen temperatura 4, paso 5 y sin fusión de posesiones
cortas.

## E18 — Pistas ampliadas (`features_v2`)

Base `e17_mix_train` más pistas solo de jugadores, tiempo desde/hasta el último avistamiento del
balón, quietud y contexto largo.

| Config | spellF1 | macroF1 | D-F1 | random35 | mixed41_art | dark50 |
|---|---|---|---|---|---|---|
| e17_mix_train | 0,6743 | 0,8841 | 0,9401 | 0,680 | 0,651 | 0,692 |
| e18_v2 | **0,6972** | **0,8915** | **0,9438** | 0,699 | 0,680 | 0,713 |

Comparación emparejada: **+0,023 de media, mejora en 10 de 10 partidos**.
**Decisión**: adoptado (también mejora con huecos reales, ver abajo).

## Validación con huecos reales (SkillCorner)

Entrenamiento con los 10 partidos externos; evaluación en 8 partidos de SkillCorner tal como
vienen (sin daño simulado). Es la única medida con pérdidas que nadie simuló.

| Modelo | spellF1 | macroF1 | D-F1 |
|---|---|---|---|
| Baseline oficial (ventana 50) | 0,3053 | | |
| e17_old_train (huecos al azar) | 0,4373 | 0,8168 | 0,8841 |
| e17_mix_train (mezcla de regímenes) | 0,4681 | 0,8412 | 0,9180 |
| e18_v2 (mezcla + pistas ampliadas) | **0,4824** | **0,8460** | **0,9294** |

**Lectura**: las dos mejoras se mantienen fuera de los regímenes simulados (+0,031 y +0,014), o
sea que no eran solo "conocer el examen". El candidato supera el 0,454 que tenía el equipo en
esta misma clase de prueba, aunque la selección de partidos puede no ser idéntica.
**Límite**: sigue habiendo una caída grande de simulado (0,70) a real (0,48). SkillCorner solo ve
4–5 jugadores por equipo, así que parte de esa caída no aplica a una cámara táctica.
