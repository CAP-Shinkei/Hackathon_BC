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

## E12 does v2 depend on the ball's visibility pattern? (`experiments/e12_visibility_ablation.py`)

v2 + fill + HSMM trained three ways: **full**; **no_seen_share** (without the rolling 0.5/2/5 s
"ball seen" shares); **no_visibility** (also without "ball seen on this frame" and the fill's
gap-shape clues). Those are the clues whose meaning comes from the simulator. Tested on E09's
simulated cross-dataset sets and on the 20 SkillCorner matches (fit on all 10 Metrica + DFL games),
with their real gaps and with simulator damage added on top. Controlled rule 2 s. spellF1:

| test | full | no_seen_share | no_visibility | baseline |
|---|---|---|---|---|
| DFL -> Metrica, sim 35% | 0.615 | **0.619** | 0.613 | 0.259 |
| DFL -> Metrica, sim 50% | **0.596** | 0.584 | 0.582 | 0.213 |
| Metrica -> DFL, sim 35% | **0.518** | 0.514 | 0.511 | 0.254 |
| Metrica -> DFL, sim 50% | **0.506** | 0.505 | 0.493 | 0.222 |
| **mean, simulated (selection number)** | **0.559** | 0.555 | 0.550 | 0.237 |
| SkillCorner, real gaps | 0.454 | 0.461 | **0.461** | 0.274 |
| SkillCorner, real + sim 35% | **0.470** | 0.468 | 0.451 | 0.214 |

On SkillCorner real gaps (full model): 236 predicted possessions vs 248 true; dead-ball share
0.439 predicted vs 0.464 true; D F1 0.90; team accuracy 0.84.

Reading:
- **v2 transfers to real gaps.** On a different league, tracker and gap pattern, never used for
  fitting, it scores 0.45 against the baseline's 0.27 (+0.18; +0.32 on simulated gaps). It does not
  over-predict dead ball there (0.44 vs 0.46 true).
- **The visibility clues are not what carries it.** Removing them costs 0.004–0.009 on simulated
  gaps and changes SkillCorner real gaps by +0.007, both within seed/game noise (~0.01). So the
  risk that v2 only works because it learned the simulator's visibility habits is small.
- SkillCorner's absolute scores are lower partly for reasons specific to broadcast data: replays
  empty 53% of dead frames of players as well as ball (n_players = 0, never seen in training), and
  its labels are SkillCorner's own automatic output.
- Decision: **keep the full model** (it wins the selection number); the ablated variants are
  not better on real gaps by more than noise.

## E13 false ball detections, centre-spot blind spot, far-outside rule (`experiments/e13_artefacts.py`)

Two artefacts seen in the competition tracking's aggregates were added to `gapsim.Params` (off by
default): bursts of false ball detections anywhere in the camera's view (pitch + 35 m), and no ball
within 3 m of the centre spot. Rule `mtah.evidence.drop_far_outside`: a ball more than 10 m beyond
the lines is treated as not seen (FIFA stadium recommendations: ~10 m of run-off around the pitch).
Models: trained on plain damage (std) or with artefacts at 5 bursts/min (art), each with and
without the rule (applied to training and test). Fit on one dataset, scored on the other.
Mean spellF1 over the four cells (DFL->Metrica, Metrica->DFL; two damage settings each):

| model | plain damage, 35% / 50% | artefacts 2 and 10 bursts/min (35%) | of which 10/min |
|---|---|---|---|
| std (current) | **0.559** | 0.521 | 0.485 |
| std + rule | **0.559** (identical) | 0.544 | 0.518 |
| art | 0.549 | 0.565 | 0.555 |
| art + rule | 0.555 | **0.568** | **0.560** |

Reading:
- **The rule is free.** Without false detections it changes nothing (identical scores); with them
  it recovers +0.02 on average and +0.03 at 10 bursts/min. Without it, false detections inside the
  pitch and far outside push the std model to under-predict dead ball (0.27 vs 0.45 true at
  10/min on DFL).
- **Training with artefacts** helps more when they are present (+0.04 to +0.07) but costs ~0.004
  without them. Whether it pays off depends on how often the competition tracking has false
  detections, which we have not measured (measuring gap statistics on the ATD is an open question
  to the organisers).
- Decision: **adopt the rule** (`make_submission.py --drop-outside`; weakly better everywhere).
  Artefact training: team decision, pending the organisers' answer.

---

**Numbering.** E14-E24 are the team's experiments on the `main` branch of
`CAP-Shinkei/Hackathon_BC` (whistle, a rebuild of this pipeline, regime-mix training, extended
features, SkillCorner training); their log is `EXPERIMENTS.md` there. This branch continues at E25.

## Where the E13 candidate lost its score (`experiments/anatomy/`, 2026-11-11)

The E13 candidate re-fitted in the E10 protocol (it reproduces E10 to the last digit), then its
predictions on the external test games compared with three partly-perfect versions of themselves.
Mean spellF1 over games:

| cell | actual | true team inside our live stretches | true dead-ball mask, our team | our possessions with perfect boundaries |
|---|---|---|---|---|
| DFL -> Metrica, 15% live missing | 0.662 | 0.903 | 0.745 | 0.779 |
| DFL -> Metrica, 35% | 0.615 | 0.882 | 0.719 | 0.756 |
| DFL -> Metrica, 50% | 0.596 | 0.851 | 0.685 | 0.734 |
| Metrica -> DFL, 15% | 0.539 | 0.899 | 0.627 | 0.691 |
| Metrica -> DFL, 35% | 0.518 | 0.885 | 0.613 | 0.676 |
| Metrica -> DFL, 50% | 0.506 | 0.880 | 0.603 | 0.671 |

- The loss is in **which team has the ball and when it changes**: +0.24 to +0.37 if the team
  were right, against +0.08 to +0.10 if dead ball were right.
- Ball gaps are not the main limit: 50% -> 15% of live frames missing gains only 0.03-0.07, and
  even with the ball visible the classifier names the right team on 87-91% of frames.
- Recall at IoU 0.5 by true possession length: under 3 s 0.13-0.19, 3-6 s 0.27-0.46, over 12 s
  0.72-0.91. Possessions under 6 s are 39-45% of all.
- Turnover timing is imprecise, not late: 35-52% of matched turnover boundaries are within 1 s of
  the truth, and moving every predicted turnover 0.2-1.2 s later gains at most 0.005.
- The controlled-possession threshold alone (`definition_cost.py`): a perfect model using 2 s
  scores 0.76-0.80 if the key uses 0 s, 0.85-0.89 for 1 s, 0.86-0.90 for 3 s, 0.69-0.76 for 5 s.
  About 0.10 per second of mismatch; 2 s is close to the safest pick over 0-3 s.

This is what pointed E25-E27 at turnover evidence rather than at the decoder or the ball fill.

## E25 one benchmark for every variant; more training regimes; context features (`experiments/e25_benchmark.py`)

One script now scores every variant of v2 + fill + HSMM + rule on the same test sets:
- **std35, std50**: E10's four cells (fit on one dataset, score on the other; seeds 7, 8). Their
  mean is the **selection number**.
- **art10**: the same with 10 false-detection bursts a minute and the centre blind spot (E13's
  hardest cell).
- **dark**: 7% of live frames missing, ball seen on a fifth of each stoppage (SkillCorner's real
  figures, E11).
- **sc_real**: fit on Metrica + DFL, score on 8 SkillCorner games (real gaps) never trained on.

`std` reproduces E10 exactly (0.6154 / 0.5958 / 0.5179 / 0.5062). Variants:
- **mix**: two more training regimes per game (`gapsim.TRAIN_EXTRA`): E13's artefacts, and the
  dark-stoppage regime. Five damaged copies per game instead of three.
- **+ctx** (`features.compute_context`, 28 clues, after the team's E18): the nearest-team share
  and the ball-seen share over the 6 s and 12 s before and after each frame separately; each
  team's closest approach to the ball in the second before and after; each team's distance to
  where the ball was last and is next seen; share of players standing still; team length, width,
  spread rate, sprinters.
- **+turn** (`features.compute_turnover`, 21 clues, E26 below).

spellF1:

| variant | DFL->Metrica 35% | 50% | Metrica->DFL 35% | 50% | **selection** | art10 | dark | sc_real |
|---|---|---|---|---|---|---|---|---|
| baseline (+/-2 s vote) | 0.259 | 0.213 | 0.254 | 0.222 | 0.237 | 0.229 | 0.355 | 0.294 |
| std (E13 candidate) | 0.615 | 0.596 | 0.518 | 0.506 | 0.559 | 0.518 | 0.626 | 0.463 |
| mix | 0.609 | 0.587 | 0.521 | 0.502 | 0.555 | 0.556 | 0.631 | 0.467 |
| mix+ctx | 0.698 | 0.662 | 0.630 | 0.620 | 0.652 | 0.649 | 0.704 | 0.469 |
| mix+turn | 0.690 | 0.650 | 0.603 | 0.578 | 0.630 | 0.618 | 0.715 | 0.487 |
| **mix+ctx+turn** | **0.720** | **0.681** | **0.656** | **0.640** | **0.674** | **0.664** | **0.734** | 0.475 |

Reading:
- **mix** costs nothing on the selection number (-0.004, inside noise) and gains +0.04 with false
  detections: AGENTS.md's old next step 2, done.
- **The context features are the largest single gain since v2: +0.097**, in every cell and under
  all three simulated gap patterns. On SkillCorner they add nothing (+0.002). Likely reason, not
  measured: several of them describe the whole team, and a broadcast camera only sees the 4-5
  players per team near the ball, while a tactical camera (ours: ~20 players tracked) sees all.
- The two families add up (0.674), and the combination is the best on every simulated cell.
- Caveat on the size of the gain: it is proven on simulated gaps in full-pitch tracking. On real
  broadcast gaps the candidate moves from 0.463 to 0.475 only.
- Not run: `mix+ctx+turn+sc` (12 SkillCorner games added to training, after the team's E24), and
  `experiments/e25_ablation.py` (which context clues carry the gain). Both need the machine alone.

Decision: **adopt mix+ctx+turn** (flags `--regimes mix --context --turnover`).

**The team's pipeline on these test sets** (`e25_team_export.py`, `e25_team_side.py`): their
code, training regimes, features and decoder unchanged, their pinned environment; only the split
and the test damage are ours. Their old-style configuration (`e17_old_train`: random gaps, their
base features): selection 0.515, art10 0.474, dark 0.622, sc_real 0.481. Their two newer
configurations (`e18_v2`, `e24_v2_real`, the one they ship) were not run: memory.

**Pipeline checks on the competition match (aggregates only; nothing was chosen by them).**
- The team's shipped bundle (`main`, md5 of its CSV 5df8f51f...) runs cleanly on the match: 232
  possessions, dead ball 45.2% of graded frames, home 60.2% of live time, no sanity warnings.
- Frame agreement / cross spellF1 between files: E13 candidate vs team 0.892 / 0.609; new
  candidate vs team 0.906 / 0.666; E13 candidate vs new candidate 0.879 / 0.563.
- The new clues have the same aggregate ranges on the match tracking as on the external games.
- The match's ball gaps are long: at least 10% of graded frames lie in a gap of more than 54 s,
  and the short-window visibility is almost all-or-nothing. That is the dark-stoppage picture of
  E11, not the default simulator's many 1 s gaps. Not used to set anything (whether simulator
  settings may be measured on the provided tracking is an open question to the organisers).

## E26 turnover features (`mtah/features.py: compute_turnover`, rows `+turn` above)

Clues aimed at the moment possession changes, which the anatomy showed is where the score goes:
how long the same team has been and stays nearest to the (filled) ball, and the length of the
runs before and after (the challenge's "keeps the ball for a few seconds" as a clue); the break
in the ball's velocity over 0.4 s before and after each frame, its largest value in the 2 s
before and after split by which team's player was nearest, and the time since and until each
team last "played" the ball (a break above 6 m/s); the closing speed of the ball on each team's
nearest player; the nearest-team share over the 1 s and 3 s before and after.

Alone on top of mix: selection +0.075, and the only family that helps on SkillCorner's real gaps
(+0.020). On top of the context features: +0.022. Decision: **keep**, with the context features.

## E27 sequence model: a small transformer over the same clues (`mtah/sequence.py`, `experiments/e27_sequence.py`, `e27_stack.py`)

A transformer (3 layers, width 96, 284k parameters) reads ~50 s of the 79 clues at 25 fps and
labels every 0.2 s. Two extra outputs are training signals only: touch-level possession, and the
hidden ball's distance to each team's nearest player on frames where the simulator hid the ball
(Manuel's "restore what was deleted" idea, in the form this model can use). Same folds, same HSMM
and the same test sets as E25; the test prediction is the mean of the three fold models.
Stacking = geometric mean with the classifier's probabilities, no fitted weight.

| model | DFL->Metrica 35% | 50% | Metrica->DFL 35% | 50% | **selection** | art10 | dark |
|---|---|---|---|---|---|---|---|
| mix+ctx (classifier) | 0.698 | 0.662 | 0.630 | 0.620 | 0.652 | 0.649 | 0.704 |
| sequence model alone | 0.717 | 0.698 | 0.618 | 0.589 | 0.656 | 0.629 | 0.732 |
| mix+ctx x sequence | 0.712 | 0.696 | 0.636 | 0.628 | 0.668 | 0.650 | 0.730 |
| mix+ctx+turn (classifier) | 0.720 | 0.681 | 0.656 | 0.640 | 0.674 | 0.664 | 0.734 |
| mix+ctx+turn x sequence | 0.724 | 0.703 | 0.643 | 0.628 | 0.675 | 0.652 | 0.741 |

Reading:
- The sequence model depends on how many games it learns from: fitted on 7 DFL games it beats
  the classifier that lacks the turnover clues by +0.03; fitted on 3 Metrica games it loses by
  0.02. Alone it is level with mix+ctx and below mix+ctx+turn.
- **Once the classifier has both feature families, stacking adds nothing**: +0.0003 on the
  selection number, -0.012 with false detections, +0.006 with dark stoppages.
- So the gain of this round came from better clues, not from the model type. The version without
  the two training-only outputs (`--no-aux`) was not run, so their own effect is unmeasured.
- Not run on SkillCorner.

Decision: **not adopted as the main candidate.** Kept as a second file at Manuel's request (see
the submissions below), with the caveat that a neural network's arithmetic can differ in the last
bits between machines, while the rules ask for the exact file.

## Submissions of 2026-11-11

**Candidate 1** (`python make_submission.py --model v2 --min-s 2 --fill --decoder hsmm --drop-outside --regimes mix --context --turnover`)

- Fit on all 10 damaged external games, five regimes each: out-of-fold external spellF1 0.7136
  (HSMM temperature 0.3). The E13 candidate at the same step: 0.5745.
- Output: 261 possessions, mean 13.2 s; graded frames H 0.326 / A 0.215 / D 0.459; home 0.602.
- Official validator: OK. md5 0f7c533b825f238699e08df437a5a590.
- With the same code the old command regenerated the E13 candidate byte for byte (cf68da6c...).

**Candidate 2** (the same command plus `--stack model/sequence_mix_ctx_turn.pt`; PyTorch)

- `python train_sequence.py` (1 h 7 min here): three fold models of the sequence model on all 10
  games; weights md5 882a2bccdb86a0f021ed432e8983e051. Out-of-fold external spellF1 on those
  games: classifier 0.7136, sequence model 0.7165, stack 0.7249 (HSMM temperature 0.5). This
  +0.011 mixes both datasets in training, unlike the selection number (+0.0003), and fits E27's
  reading that the sequence model gets better with more games.
- Output: 282 possessions, mean 12.1 s; graded frames H 0.324 / A 0.212 / D 0.464; home 0.605.
- Official validator: OK. md5 a0379735c47768f1c5867e531604c049.
- Against candidate 1 on the match: 96.6% of graded frames agree, cross spellF1 0.829.
- Reproducibility: with one thread (which `make_submission.py` sets) the sequence model's output
  on the match is bit-identical from run to run on this machine; with six threads it differs by
  up to 2e-8. Another CPU may differ in the same way, and whether a difference of that size can
  move a label was not tested. The file was generated once; it was not regenerated end to end a
  second time. Candidate 1 has none of this risk.

Which to submit: the selection number does not separate them (0.6742 / 0.6745). Candidate 1 is
the recommendation because the rules ask for the exact file; candidate 2 is the alternative if
the team prefers the out-of-fold +0.011.
