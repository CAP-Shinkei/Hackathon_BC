# Experiment log

Selection rule (fixed 2026-11-09, before any experiment): versions are compared by **mean spellF1
over the external games**, gap-damaged by `external/gapsim.py`, leave-one-game-out whenever
something is fitted. The public 5 minutes of the competition match are a pipeline check only.

External games: Metrica sample games 1-3, answer keys from `external/build_gt.py`.
Scores come from the official metric code (`mtah/scoring.py` imports `possession_metrics.py`).

## E01 baseline on clean external games (`experiments/e01_baseline_clean.py`)

Official baseline method (nearest player + majority vote), mean over the 3 games:

| window | spellF1 | macroF1 |
|---|---|---|
| raw | 0.107 | 0.849 |
| +/-1 s | 0.366 | 0.858 |
| +/-2 s | 0.415 | 0.842 |
| +/-5 s | 0.300 | 0.756 |

Clean data flatters it: D F1 is 0.88-0.98, because in cleaned tracking a missing ball means a dead ball.

## E02 baseline on gap-damaged external games (`experiments/e02_baseline_damaged.py`)

Simulator parameters set by physical reasoning, swept over the share of live frames without
ball. Not fitted to the competition match. Mean spellF1 over the 3 games:

| setting | raw | +/-1 s | +/-2 s | +/-5 s |
|---|---|---|---|---|
| clean | 0.107 | 0.366 | 0.415 | 0.300 |
| 15% live missing | 0.038 | 0.227 | 0.306 | 0.272 |
| 35% live missing | 0.026 | 0.175 | 0.255 | 0.262 |
| 50% live missing | 0.024 | 0.138 | 0.209 | 0.240 |
| 35%, ball never seen while dead | 0.033 | 0.198 | 0.286 | 0.270 |

## E03 v1 = HMM + Viterbi + merge (`experiments/e03_hmm_v1.py`)

Team-symmetric emission and transition tables counted on the two training games (3 gap rates
each). Temperature and minimum spell length are chosen on the training games and tested once on
the held-out game (different damage seed).
Chosen: temperature 0.2, minimum spell 50-75 frames.

| test setting | v1 spellF1 | baseline +/-2 s | v1 macroF1 |
|---|---|---|---|
| 15% live missing | **0.388** | 0.311 | 0.714 |
| 35% live missing | **0.308** | 0.257 | 0.619 |
| 50% live missing | **0.235** | 0.222 | 0.511 |

Weakness: v1 predicts ~190-300 possessions where the keys have ~360-410, so it over-merges.

## E04 kick-off detector (`experiments/e04_kickoff_external.py`)

Ball leaving the centre spot, inside a window of a few seconds around the true kick-off:
80-85% detected, median error 4-5 frames. On the competition match it finds nothing (the ball is
never tracked within 3 m of the spot in that window), so it falls back to the Smart Tagging
phase start, 82 frames before the published kick-off.
Rejected alternative: a players-based detector (kick-off = first frame with 2 players more than 1 m
inside the opponents' half). On the external games it was 20 frames late at the median and more
than 120 frames off in 10% of cases, which is worse than the tag fallback.

## Submission v1 (`python make_submission.py`, 2026-11-09)

- Fit on all 3 damaged external games: temperature 0.2, minimum spell 50 frames, external spellF1 0.302.
- Output: 334 possessions, mean 10.8 s; graded frames H 0.333 / A 0.232 / D 0.435.
- Official validator: OK. Re-running the script gives a byte-identical file (md5 3d8c9402...).
- Pipeline check on the public 5 minutes, **not used for any choice**: 19 possessions
  (truth 17), mean 13.9 s (truth 14.2 s).

## E05 v1 across datasets and possession definitions (`experiments/e05_cross_dataset.py`)

Added the IDSSE/DFL open dataset (7 matches, per-frame BallPossession/BallStatus). v1 is fitted
on one dataset and scored once on the other. "def" is the controlled-possession rule applied to the
external labels (`external/controlled.py`): 0 = touch-level as shipped; 2 or 3 = spells shorter than
that many seconds go back to the previous team. Mean spellF1 (v1 / baseline +/-2 s):

| def | train -> test | 15% missing | 35% missing | 50% missing | v1 spells / true spells (35%) |
|---|---|---|---|---|---|
| 0 | DFL -> Metrica | 0.391 / 0.311 | 0.308 / 0.258 | 0.205 / 0.198 | 245 / 384 |
| 0 | Metrica -> DFL | 0.313 / 0.278 | 0.250 / 0.246 | 0.174 / 0.213 | 251 / 437 |
| 2 | DFL -> Metrica | 0.476 / 0.327 | 0.379 / 0.267 | 0.247 / 0.202 | 207 / 270 |
| 2 | Metrica -> DFL | 0.392 / 0.305 | 0.314 / 0.260 | 0.220 / 0.218 | 215 / 288 |
| 3 | DFL -> Metrica | 0.499 / 0.315 | 0.399 / 0.246 | 0.254 / 0.187 | 204 / 232 |
| 3 | Metrica -> DFL | 0.413 / 0.285 | 0.338 / 0.238 | 0.227 / 0.199 | 192 / 237 |

v1 transfers across datasets and beats the baseline in 16 of 18 cells; it loses only at 50% missing
with touch-level labels (Metrica -> DFL). Scores under different definitions are not comparable with
each other (the target changes); the definition is a decision, not something to optimise.

## E06 v2 = feature classifier + hybrid HMM, across datasets (`experiments/e06_v2_cross_dataset.py 0 2 3`)

Per-frame features (`mtah/features.py`): distances to the nearest player of each team, ball
outside the pitch and distance to the lines, ball speed, each team's mean speed and spread, number
of tracked players, and rolling 0.5 s / 2 s / 5 s shares of "nearest team is home", "ball seen" and
mean player speed. A gradient-boosted classifier, trained team-symmetric, gives P(H/A/D); the
same Viterbi as v1 decodes it. Temperature and minimum spell are chosen on out-of-fold predictions.
Fit on one dataset, scored once on the other. Mean spellF1 (v2 / v1 / baseline +/-2 s):

| def | train -> test | 15% missing | 35% missing | 50% missing | v2 spells / true (35%) |
|---|---|---|---|---|---|
| 0 | DFL -> Metrica | **0.556** / 0.391 / 0.311 | **0.516** / 0.308 / 0.258 | **0.479** / 0.205 / 0.198 | 298 / 384 |
| 0 | Metrica -> DFL | **0.409** / 0.313 / 0.278 | **0.401** / 0.250 / 0.246 | **0.375** / 0.174 / 0.213 | 255 / 437 |
| 2 | DFL -> Metrica | **0.648** / 0.476 / 0.327 | **0.619** / 0.379 / 0.267 | **0.574** / 0.247 / 0.202 | 240 / 270 |
| 2 | Metrica -> DFL | **0.523** / 0.392 / 0.305 | **0.520** / 0.314 / 0.260 | **0.495** / 0.220 / 0.218 | 221 / 288 |
| 3 | DFL -> Metrica | **0.680** / 0.499 / 0.315 | **0.641** / 0.399 / 0.246 | **0.627** / 0.254 / 0.187 | 202 / 232 |
| 3 | Metrica -> DFL | **0.577** / 0.413 / 0.285 | **0.567** / 0.338 / 0.238 | **0.542** / 0.227 / 0.199 | 212 / 237 |

v2 wins all 18 cells and degrades much less as the ball goes missing (it does not depend on the
ball alone). Caveat: some features read the ball's visibility pattern, which comes from the
simulator; if real gaps differ from simulated ones, part of this gain may not carry over. Real gap
patterns (SkillCorner) are the next check.

## E07 filling the missing ball before v2 (`experiments/e07_ballfill.py`)

Ball gaps are filled by straight-line interpolation between the last and next sighting
(`mtah/ballfill.py`), optionally plus a learned correction (gradient boosting on the gap shape and the
players around the point, trained on the training dataset's clean ball). Filled-ball clues are
added to v2's features: nearest players to the filled ball, time since and until the ball is seen.
Controlled rule 2 s; fit on one dataset, scored once on the other. spellF1 (median fill error):

| train -> test | missing | v2 | v2 + straight-line fill | v2 + learned fill |
|---|---|---|---|---|
| DFL -> Metrica | 35% | 0.604 | **0.612** (3.5 m) | 0.607 (3.4 m) |
| DFL -> Metrica | 50% | 0.561 | 0.582 (4.3 m) | **0.585** (4.2 m) |
| Metrica -> DFL | 35% | 0.505 | **0.514** (3.3 m) | 0.512 (3.4 m) |
| Metrica -> DFL | 50% | 0.492 | **0.501** (4.3 m) | 0.500 (4.3 m) |

Filling helps a little in all four cells (+0.008 to +0.024), more when more ball is missing. The
learned correction barely moves the fill error (3.5 -> 3.4 m at best, worse in one direction) and
gives no extra spellF1: the players around a hidden ball do not say much more about where it is
than the straight line does. Keep: straight-line fill (cheap, no training). Drop: learned correction.

## E09 ball fill x decoder (`experiments/e09_fill_hsmm.py`)

The four combinations of {no fill, straight-line fill} x {Viterbi + merge, duration-aware HSMM}.
The HSMM (`mtah/hsmm.py`, after Álvaro's v0.3) uses duration distributions and spell transitions
counted on the training labels; its temperature is chosen out-of-fold. Controlled rule 2 s; fit on
one dataset, scored once on the other. spellF1:

| train -> test | missing | v2 | v2 + HSMM | v2 + fill | v2 + fill + HSMM |
|---|---|---|---|---|---|
| DFL -> Metrica | 35% | 0.604 | 0.609 | 0.612 | **0.615** |
| DFL -> Metrica | 50% | 0.561 | 0.566 | 0.582 | **0.596** |
| Metrica -> DFL | 35% | 0.505 | 0.507 | 0.514 | **0.518** |
| Metrica -> DFL | 50% | 0.492 | 0.493 | 0.501 | **0.506** |

Both additions help in all four cells and they stack: fill +0.008 to +0.021, HSMM +0.001 to
+0.014, together +0.011 to +0.035. Best version: v2 + fill + HSMM.

## E10 every version on the same held-out games (`experiments/e10_all_versions.py` + E07/E09)

All rows use identical test sets: the other dataset, damaged with seeds 7/8, controlled rule 2 s.
Álvaro's versions use his own constants but **no Smart Tagging bonus** (the external games have no
Smart Tagging), so they measure his smoothing designs, not his full submission. spellF1:

| version | what it is | DFL->Metrica 35% | 50% | Metrica->DFL 35% | 50% | mean |
|---|---|---|---|---|---|---|
| baseline | organisers: nearest player + vote +/-2 s | 0.259 | 0.213 | 0.254 | 0.222 | 0.237 |
| Álvaro v0.2 | his Viterbi + 3.2 s merge, his constants | 0.297 | 0.190 | 0.264 | 0.178 | 0.232 |
| Álvaro v0.3 | his HSMM, his gamma durations | 0.291 | 0.228 | 0.287 | 0.221 | 0.257 |
| v1 | v0.2 design, numbers refitted on external data | 0.369 | 0.238 | 0.313 | 0.216 | 0.284 |
| v2 | trained classifier on ~25 per-frame clues + Viterbi | 0.604 | 0.561 | 0.505 | 0.492 | 0.541 |
| v2 + HSMM | duration-aware smoothing instead of Viterbi + merge | 0.609 | 0.566 | 0.507 | 0.493 | 0.544 |
| v2 + learned fill | + clues from a trained ball-position fill | 0.607 | 0.585 | 0.512 | 0.500 | 0.551 |
| v2 + fill | + clues from a straight-line ball fill | 0.612 | 0.582 | 0.514 | 0.501 | 0.552 |
| **v2 + fill + HSMM** | both | **0.615** | **0.596** | **0.518** | **0.506** | **0.559** |

Reading: Álvaro's designs with his own numbers are at baseline level on unseen games (v0.2 is
below the baseline when half the ball is missing). Refitting the numbers on external data (v1)
adds about +0.03 to +0.05. The big step is the trained per-frame evidence (v2), +0.26 over v1; fill and HSMM add
a further +0.02 together.

## Submission v2 + fill + HSMM (`python make_submission.py --model v2 --min-s 2 --fill --decoder hsmm`)

- Fit on all 10 damaged external games: classifier temperature/minimum spell chosen out-of-fold,
  HSMM temperature 0.3, out-of-fold external spellF1 0.575.
- Output: 259 possessions, mean 12.7 s; graded frames H 0.313 / A 0.205 / D 0.482; home 0.605.
- Official validator: OK. md5 856330e93c05226c31509c68ef19be15.
- Before generating it, the edited script reproduced `v2_min2s.csv` byte for byte (md5 47e7cc28...).

## E11 real ball gaps (SkillCorner) against the gap simulator (`experiments/e11_skillcorner_gaps.py`)

SkillCorner open data (20 A-League 2024/25 matches, MIT; loader `external/skillcorner.py`) is
automatic broadcast tracking with its real ball gaps: each ball frame says whether it was detected
or extrapolated (extrapolated = treated as missing), and replays leave frames empty. Live play =
inside SkillCorner's phases of play; labels are SkillCorner's own automatic output. Upsampled
10 -> 25 fps. The same aggregates on the simulator's damage of the 10 Metrica + DFL games. Means
over games:

| | SkillCorner real | gapsim 35% | gapsim 50% |
|---|---|---|---|
| ball missing, graded frames | 0.407 | 0.451 | 0.532 |
| ball missing, live frames | **0.073** | 0.351 | 0.500 |
| ball seen, dead frames | 0.208 | 0.414 | 0.425 |
| frames with no data at all (replays), live / dead | 0.017 / 0.533 | - | - |
| live gaps per minute of live play | 2.1 | 10.7 | 13.1 |
| live gap length, median / 90th pct | 1.18 s / 4.50 s | 1.03 s / 4.69 s | 1.18 s / 5.62 s |
| share of live gap time in gaps > 5 s | 0.386 | 0.390 | 0.468 |
| gap-start hazard vs average: 0 / 1 / 2+ players within 2 m | **1.33 / 0.78 / 0.82** | 0.89 / 1.01 / 1.47 | 0.87 / 1.02 / 1.52 |
| ... ball slower than 5 m/s / faster than 15 m/s | 0.93 / 1.20 | 0.94 / 1.07 | 0.99 / 1.04 |
| ... ball below 1 m / above 2 m | **0.73 / 2.71** | - | - |
| ball seen in first / last quarter of dead stretches > 4 s | **0.43 / 0.32** | 0.16 / 0.76 | 0.16 / 0.77 |

Reading:
- **Gap lengths match.** The simulator's lognormal (median 1 s, sigma 1) reproduces the real
  median, 90th percentile and long-gap share closely.
- **The rate does not transfer.** A broadcast camera follows the ball, so SkillCorner loses it on
  only 7% of live frames; total missing on graded frames (0.41) is nonetheless almost exactly the
  competition's published 41%, because the ball is unseen during 79% of dead time (replays). The
  competition's single 41% figure cannot tell "few live gaps, dark stoppages" from "many live gaps,
  visible stoppages", and the simulator assumes the second.
- **What starts a gap differs.** Real gaps start when the ball is in the air (2.7x above 2 m) and
  slightly less often near players; the simulator makes crowding the main cause. The official
  DATA.md says crowded boxes *and* balls in the air both lose the ball in the competition tracking,
  so the broadcast pattern is not proof the simulator is wrong, but height is a missing mechanism
  (DFL tracking has ball height, Metrica does not).
- **Dead-ball visibility is reversed.** The simulator shows the ball in a tail at the restart; in
  SkillCorner it is seen more at the start of a stoppage (before the replay cut). Broadcast replays
  do not exist in a fixed tactical camera, so this part of SkillCorner is not representative either.
- Side finding: the simulator's 0.3 m per-frame jitter makes the measured ball speed noisy
  (about 10 m/s of noise at 25 fps), which is why its "fast ball" effect almost disappears above.
