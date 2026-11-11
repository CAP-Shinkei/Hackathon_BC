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
tracking ──► gap-damaged external games ──► 79 per-frame clues ──► classifier P(H,A,D)
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
  training. Each game is damaged five ways (`--regimes mix`, E25): live-gap share 15/35/50%,
  plus false detections with the centre-spot blind spot (E13), plus dark stoppages with few live
  gaps (SkillCorner's real figures, E11).
- **Clues** (`mtah/features.py`, `mtah/ballfill.py`): 30 base clues (distances ball→nearest
  home/away player, ball outside the pitch, speeds, team spreads, number of players, rolling
  0.5/2/5 s shares; the same distances on a straight-line-filled ball, and the gap shape), 28
  context clues (`--context`, E25: one-sided 6 s / 12 s windows before and after, each team
  against the ball's last and next sighting, stillness, team shape) and 21 turnover clues
  (`--turnover`, E26: runs of one team nearest the ball, breaks in the ball's velocity and who
  is there, closing speeds).
- **Classifier** (`mtah/v2.py`): `HistGradientBoostingClassifier`, team-symmetric (each frame also
  added with home and away swapped). Temperature and minimum spell chosen out-of-fold.
- **Smoothing** (`mtah/hsmm.py`): HSMM with duration distributions counted on the training labels.
- **Rules** (`make_submission.py`, `mtah/kickoff.py`): per-half decoding; D before the 2nd-half
  kick-off (detected, else the Smart Tagging kick-off phase start).
- **Optional sequence model** (`mtah/sequence.py`, `train_sequence.py`, `--stack`; E27): a small
  transformer over the same clues, stacked on the classifier. It adds nothing measurable; it
  exists as a second candidate file only. Needs PyTorch (`requirements-sequence.txt`).

Older pieces kept for comparison: `mtah/v1.py` + `mtah/hmm.py` (HMM on the nearest-player rule),
`mtah/smooth.py` (the official baseline's vote).

## 4. Current state (2026-11-11)

Selection number = mean held-out spellF1 over E10's four cells (`experiments/e25_benchmark.py`).

| version | selection | false detections | dark stoppages | SkillCorner real gaps |
|---|---|---|---|---|
| organisers' baseline | 0.237 | 0.229 | 0.355 | 0.294 |
| v1 | 0.284 | | | |
| v2 + fill + HSMM + rule (E13 candidate) | 0.559 | 0.518 | 0.626 | 0.463 |
| + five training regimes (mix) | 0.555 | 0.556 | 0.631 | 0.467 |
| + context clues | 0.652 | 0.649 | 0.704 | 0.469 |
| **+ turnover clues (candidate 1)** | **0.674** | **0.664** | **0.734** | 0.475 |
| candidate 1 stacked with the sequence model (candidate 2) | 0.675 | 0.652 | 0.741 | not run |

**Candidate 1**: `out/submissions/v2_min2s_fill_hsmm_rule_mix_ctx_turn.csv`, md5
`0f7c533b825f238699e08df437a5a590`, official validator OK (261 possessions, mean 13.2 s, dead ball
46% of graded frames, home 60.2% of possession time). Regenerate (about 30 minutes):
`python make_submission.py --model v2 --min-s 2 --fill --decoder hsmm --drop-outside --regimes mix --context --turnover`.

**Candidate 2** (the same plus `--stack model/sequence_mix_ctx_turn.pt`, weights written by
`train_sequence.py`): `out/submissions/v2_min2s_fill_hsmm_rule_mix_ctx_turn_nn.csv`, md5
`a0379735c47768f1c5867e531604c049`, validator OK (282 possessions, mean 12.1 s). On the selection
number it equals candidate 1 (+0.0003); out-of-fold over all 10 games it is +0.011. A neural
network's arithmetic can differ in the last bits between machines (measured: 2e-8 between one
and six threads), and the rules ask for the exact file, so candidate 1 is the one to submit
unless the team decides otherwise.

Previous candidate: `v2_min2s_fill_hsmm_rule.csv`, md5 `cf68da6c29f3bad45d58fdcada833a09`; the
old command still regenerates it byte for byte with the current code (checked 2026-11-11).

What 2026-11-11 established (details in `EXPERIMENTS.md`):
- The E13 candidate lost its score on **which team has the ball and when it changes**, not on
  dead ball and not mainly on ball gaps. Clues aimed at that gave +0.115.
- A transformer over the same clues adds nothing once the classifier has those clues (E27).
- The controlled-possession threshold is worth about 0.10 spellF1 per second of mismatch with
  the answer key: still the largest open decision (`--min-s`, now 2 s).
- The gain is proven on simulated gaps in full-pitch tracking; on SkillCorner's real broadcast
  gaps it is +0.013 only. In aggregate the match's ball gaps are long (10% of graded frames in
  gaps over 54 s), like the dark-stoppage regime; nothing was set from that.

**The team repository.** `main` of `CAP-Shinkei/Hackathon_BC` is a teammate's independent
rebuild of this pipeline (no shared git history; their experiments are E14-E24). This code lives
on the branch `manuel-v2` there. Their numbers (0.691 / 0.596) use another protocol and are not
comparable with ours; on our test sets their old-style configuration scores 0.515, and their two
newer ones have not been run (`experiments/e25_team_side.py`). Their shipped model and ours agree
on 91% of the match's graded frames. The team must end with one public repository that
regenerates one file.

## 5. How to add an experiment

1. For a new training regime or feature family, add a variant to
   `experiments/e25_benchmark.py` (`python experiments/e25_benchmark.py std mix+ctx+turn ...`;
   finished runs are kept in `out/e25/` and skipped). Otherwise copy the closest script to
   `experiments/eNN_<name>.py` and reuse `common.py` and `e25_benchmark.py` (`featurise`, `fit`,
   `evaluate`) instead of rewriting them. Number from E28; E14-E24 are taken by the team's `main`.
2. Use **the same test sets** as every earlier entry, or the numbers are not comparable:
   fit on DFL → score on Metrica, and Metrica → DFL; test damage `Params(live_missing=r)` with
   r in (0.35, 0.5) and seeds 7, 8 (`seed0=7`); training damage `v1.GAP_RATES` with seed0=100;
   labels `controlled(gt, 2.0)`. Score with `mtah.scoring.score` (it calls the official metric code).
3. Report the mean over the four cells (the selection number), and the art10, dark and sc_real
   cells of E25 as robustness checks. Differences under ~0.01 on this benchmark are within seed
   noise; on the single real match, EVALUATION.md says ~0.04.
4. Add an `## ENN` section to `EXPERIMENTS.md`: what changed, a table, a short reading, and the
   decision (keep / drop). Keep a change only if it wins on the external games.
5. If it wins, put it behind a flag in `make_submission.py` so older files still regenerate
   byte for byte, regenerate, validate, and record the new md5.

## 6. Running things

Setup on a new machine. The code finds everything relative to a team folder that holds this
repository next to the data:

```
team_folder/
  repo/            this repository
  data/            the competition bundle from the organisers' Drive (tactical/BAR-ATH_tactical_*)
  official_repo/   git clone https://github.com/metrica-sports/more-than-a-hack-2027 official_repo
  external_data/   created by: bash repo/external/fetch_data.sh --with-skillcorner  (~4.5 GB)
```

Then `pip install -r requirements.txt` and check with candidate 1's command of section 4: the
output's md5 must be the one given there. Elsewhere, point `MTAH_DATA_DIR` and `MTAH_OFFICIAL_REPO` at the data.

- `pip install -r requirements.txt` (Python 3.12). `bash external/fetch_data.sh` fetches the open
  data into `../external_data` (`--with-skillcorner` for the experiment-only dataset).
- Environment: `MTAH_DATA_DIR` (competition bundle), `MTAH_OFFICIAL_REPO` (official repo clone,
  for the scorer and validator). Defaults: `../data`, `../official_repo`.
- Each experiment needs 20–60 minutes and up to **13 GB** of RAM with all 79 clues. On the 30 GB
  machine run **one at a time**: on 2026-11-11 three at once ran the machine out of memory and
  the kernel killed the editor and the session. Start long jobs detached (`setsid nohup ... &`)
  with output to `out/<name>.log`, so they survive a session restart.
- Caches: `out/cache/*.npz` (parsed tracking). Delete a file there if you change its loader.
- `out/` is git-ignored; submissions live in `out/submissions/`.
- Quirk: the team's USB drive has an antivirus that briefly blocks just-written files
  (`Operation not permitted`); wait a few seconds and retry.

## 7. Next steps to improve (in order of expected value)

Context: two rounds of better per-frame evidence gave the big gains (v1 0.28 -> v2 0.54 -> clues
0.67); smoothing tweaks, a constant timing shift and a transformer on the same clues gave nothing.
Differences under ~0.04 are noise on the real match, so prefer ideas that could move the external
benchmark by several hundredths, or that cut a risk.

1. **Team decision first: the controlled-possession threshold** (`--min-s`, now 2 s). Worth about
   0.10 per second of mismatch (anatomy entry of `EXPERIMENTS.md`). CHALLENGE.md calls the public
   5 minutes "the reference" for the definition and the organisers allow inspecting them to
   understand it, but tuning on them is forbidden: a person (never a hosted AI) may look, and
   whether that may inform the threshold is a question for the organisers. Until answered,
   assume no.
2. **Send the organisers' questions** (`../ORGANISER_QUESTIONS_DRAFT.md`), adding: may anything be
   fitted without labels on the provided tracking and tagging (RULES.md says yes, the later email
   says training must use external data only); may the public window inform the threshold.
   Until answered, assume no to both.
3. **Finish the E25 table**: the team's `e18_v2` and `e24_v2_real` on our test sets
   (`e25_team_side.py`, their environment), `mix+ctx+turn+sc` (SkillCorner games in training),
   and `e25_ablation.py`. Each needs the machine alone for 40-60 minutes. They decide which
   pipeline the team submits and how far to trust the context clues.
4. **Why the clues do not help on SkillCorner** (+0.013 against +0.115 simulated). Test the
   explanation that team-wide clues fail when a broadcast camera sees 4-5 players per team: hide
   players outside a moving window in the Metrica/DFL test copies and see whether the gain goes.
5. **Broadcast tracking as a second ball track.** The organisers provide the TV recording's own
   tracking (not on this machine: `data/broadcast/` is empty; only its ATD txt/xml and homography
   csv are needed, not the video). Aligned to the tactical timeline it could fill ball gaps, and
   camera cut-aways are a dead-ball clue that SkillCorner can calibrate. By the gap-rate
   sensitivity of the benchmark, worth about +0.01 to +0.04; 1-1.5 days.
6. **Ball height in the simulator.** E11: real gaps start 2.7x more often when the ball is above
   2 m. DFL positions have Z (`dfl.py` reads only X, Y today; delete the dfl caches after changing
   it). Metrica has no Z.
7. **Smart Tagging as evidence without fitted weights.** No external game has it, so nothing can
   calibrate a weight; only weight-free uses, argued from football.
8. **More training games** (PFF FC World Cup 2022: terms not found yet). The sequence model was
   better with 7 training games than with 3 (E27), so more games is what would make it worth
   revisiting, together with raw player and ball positions as its input instead of the clues.

Done or closed: training over a mix of gap regimes (E25, adopted); a sequence model over the
clues (E27, no gain); decoder tuning and timing shifts (no gain); artefact training (part of the
mix). Parked: whistle detector; a ball detector on the video.
