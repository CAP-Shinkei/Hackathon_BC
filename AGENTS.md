# Guide for coding agents (and humans) working on this repository

Read this first, then `EXPERIMENTS.md`. This file explains the rules, the code, the current state
and how to add an experiment without breaking the comparison or the competition rules.

## 1. The task in one paragraph

More Than A Hack 2027 (Metrica Sports, FC Barcelona). For each of the 199,319 frames of the
BAR-ATH tactical video, write H (Barça possession), A (Athletic), D (dead ball) or X (not graded).
Ranked by **spellF1** (possessions matched by overlap, thresholds 0.3–0.7), ties broken by macroF1.
Organisers' baseline: spellF1 0.2433 on the match. Deadline **2026-11-17** (Google Form; the last
response counts). Deliverables: the CSV, this repository (it must regenerate the exact file), and a
3–5 page technical document. Official repo: github.com/metrica-sports/more-than-a-hack-2027.

## 2. Rules you must not break (a violation disqualifies the team)

1. **Never send the competition data to a hosted AI service, and that includes you.** Do not
   `cat`, `head`, print, plot or view anything from the data bundle (`MTAH_DATA_DIR`: ATD tracking,
   Smart Tagging, homography, video), and do not print per-frame rows of our predictions on the
   match. Scripts that touch the match print **aggregates only** (counts, shares, scores,
   durations). Before running a command on the match data, ask whether its output could contain
   raw data; if so, rewrite it to aggregate.
2. **Nothing is tuned on the match.** Not on the private ground truth (we don't have it), not on
   the public 5 minutes, not on the video, not on hand labels. The public 5 minutes are a pipeline
   check only (file format, rough possession count), never a reason to pick a version or a constant.
3. **Every constant has a written source:** an external game, physics, or a football rule. Write
   the source in a comment next to the constant.
4. **Versions are chosen by one number only** (fixed before any experiment): mean spellF1 on
   external games with simulated ball gaps, fitted on one dataset and scored on the other (or
   leave-one-game-out). See section 5.

External open data (Metrica sample, IDSSE/DFL, SkillCorner) is not restricted: read it freely.

## 3. How the method works

```
tracking ──► gap-damaged external games ──► 30 per-frame clues ──► classifier P(H,A,D)
                                                                       │
                     duration-aware smoothing (HSMM) ◄─────────────────┘
                                   │
                     fixed rules (per half, dead before 2nd-half kick-off) ──► CSV
```

- **Training data** (`external/`): Metrica sample games (3; answer keys from events,
  `build_gt.py`) and IDSSE/DFL (7; per-frame possession and ball status, `dfl.py`). Touch-level
  possession becomes *controlled* possession with `controlled.py` (spells < 2 s go back to the
  previous team; the 2 s is a team decision read from CHALLENGE.md, not fitted).
- **Gap simulator** (`external/gapsim.py`): the open games have the ball on almost every frame,
  the match tracking loses it on 41% of graded frames, so ball positions are deleted before
  training (live-gap share swept 15/35/50%). Optional artefacts (false detections, centre-spot
  blind spot) are off by default.
- **Clues** (`mtah/features.py`, `mtah/ballfill.py`): distances ball→nearest home/away player,
  ball outside the pitch, speeds, team spreads, number of players, rolling 0.5/2/5 s shares; the
  same distances on a straight-line-filled ball, and the gap shape.
- **Classifier** (`mtah/v2.py`): `HistGradientBoostingClassifier`, team-symmetric (each frame also
  added with home and away swapped). Temperature and minimum spell chosen out-of-fold.
- **Smoothing** (`mtah/hsmm.py`): HSMM with duration distributions counted on the training labels.
- **Rules** (`make_submission.py`, `mtah/kickoff.py`): per-half decoding; D before the 2nd-half
  kick-off (detected, else the Smart Tagging kick-off phase start).

Older pieces kept for comparison: `mtah/v1.py` + `mtah/hmm.py` (HMM on the nearest-player rule),
`mtah/smooth.py` (the official baseline's vote).

## 4. Current state (2026-11-09)

| version | mean held-out spellF1 (E10) |
|---|---|
| organisers' baseline | 0.237 |
| Álvaro v0.2 / v0.3 with his constants, no Smart Tagging | 0.232 / 0.257 |
| v1 | 0.284 |
| v2 | 0.541 |
| **v2 + fill + HSMM (candidate)** | **0.559** |

Candidate file: `out/submissions/v2_min2s_fill_hsmm.csv`, md5 `856330e93c05226c31509c68ef19be15`,
official validator OK. Regenerate: `python make_submission.py --model v2 --min-s 2 --fill --decoder hsmm`.

Main open risk: the gaps are simulated. E11 compared the simulator with real gaps (SkillCorner);
E12/E13 test robustness. Read their entries in `EXPERIMENTS.md` before changing the simulator.

## 5. How to add an experiment

1. Copy the closest script in `experiments/` to `experiments/eNN_<name>.py`. Reuse
   `common.py` (loaders, `table`) and `e12_visibility_ablation.py` (`featurise`, `fit`,
   `evaluate`) instead of rewriting them.
2. Use **the same test sets** as every earlier entry, or the numbers are not comparable:
   fit on DFL → score on Metrica, and Metrica → DFL; test damage `Params(live_missing=r)` with
   r in (0.35, 0.5) and seeds 7, 8 (`seed0=7`); training damage `v1.GAP_RATES` with seed0=100;
   labels `controlled(gt, 2.0)`. Score with `mtah.scoring.score` (it calls the official metric code).
3. Report the mean over the four cells. Differences under ~0.01 on this benchmark are within
   seed noise; on the single real match, EVALUATION.md says ~0.04.
4. Add an `## ENN` section to `EXPERIMENTS.md`: what changed, a table, a short reading, and the
   decision (keep / drop). Keep a change only if it wins on the external games.
5. If it wins, put it behind a flag in `make_submission.py` so older files still regenerate
   byte for byte, regenerate, validate, and record the new md5.

## 6. Running things

- `pip install -r requirements.txt` (Python 3.12). `bash external/fetch_data.sh` fetches the open
  data into `../external_data` (`--with-skillcorner` for the experiment-only dataset).
- Environment: `MTAH_DATA_DIR` (competition bundle), `MTAH_OFFICIAL_REPO` (official repo clone,
  for the scorer and validator). Defaults: `../data`, `../official_repo`.
- Each experiment needs ~3–7 GB of RAM and 20–60 minutes; on a 30 GB machine run one at a time.
  Run long jobs in the background with output to `out/<name>.log`.
- Caches: `out/cache/*.npz` (parsed tracking). Delete a file there if you change its loader.
- `out/` is git-ignored; submissions live in `out/submissions/`.
- Quirk: the team's USB drive has an antivirus that briefly blocks just-written files
  (`Operation not permitted`); wait a few seconds and retry.

## 7. Ideas not yet done (rough order of expected value)

See `EXPERIMENTS.md` (latest entries) and the team's `PLAN.md`. Main ones: train over a *mix* of
gap patterns (including "few live gaps, dark stoppages", as in SkillCorner) so the model does not
depend on one simulator setting; a ball-height gap mechanism (DFL has ball height); a small
sequence model (GRU) instead of rolling windows; more training games (PFF World Cup 2022, terms
first); Smart Tagging as evidence without fitted weights.
