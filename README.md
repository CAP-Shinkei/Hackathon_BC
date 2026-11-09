# MTAH 2027: frame-level possession from automatic tracking

Our entry to the More Than A Hack 2027 Initial Challenge (Metrica Sports, FC Barcelona). It labels
every frame of the BAR-ATH tactical video as home possession (H), away possession (A) or dead ball
(D), and writes the `frame,label` CSV the challenge asks for.

No number in the pipeline comes from the competition match. Every model and constant is fitted or
chosen on **external** open datasets, after deleting ball positions there so that they look like the
competition's automatic tracking (`external/gapsim.py`). The competition data is only read at the
very end, as input.

## How it works

1. **Training data.** Two open datasets with frame-level truth: the Metrica sample games (3 games,
   answer keys built from their events by `external/build_gt.py`) and the IDSSE/DFL dataset
   (7 Bundesliga games with per-frame possession and ball status). Touch-level possession is turned
   into *controlled* possession: a spell shorter than 2 s goes back to the team that had the ball
   (`external/controlled.py`). Ball gaps are simulated at 15%, 35% and 50% of live frames.
2. **Evidence.** 30 per-frame clues from the tracking alone (`mtah/features.py`): distance
   from the ball to the nearest player of each team, ball position and speed, team speeds and
   spreads, rolling shares over 0.5 s / 2 s / 5 s, plus the same clues on a ball filled in by
   straight lines across its gaps (`mtah/ballfill.py`). A gradient-boosted classifier, trained
   team-symmetric, turns them into P(H), P(A), P(D) per frame (`mtah/v2.py`).
3. **Smoothing.** A duration-aware hidden semi-Markov model (`mtah/hsmm.py`) turns the per-frame
   probabilities into possessions. Its duration distributions and switch rates are counted on the
   training labels; its temperature is chosen out-of-fold.
4. **Fixed rules.** Each half is decoded separately; the second half is dead ball until its
   kick-off, found from the ball leaving the centre spot or, failing that, the start of the
   Smart Tagging kick-off phase (`mtah/kickoff.py`).

Every version and its scores on held-out external games are in `EXPERIMENTS.md`.

## Reproduce the submission

Python 3.12, about 8 GB of RAM, about 20 minutes on 14 cores.

```bash
pip install -r requirements.txt

# External training data next to this repository (../external_data), ~2.7 GB,
# and the Metrica answer keys:
bash external/fetch_data.sh

# Competition data bundle (not redistributable, not in this repository):
export MTAH_DATA_DIR=/path/to/data            # contains tactical/BAR-ATH_tactical_atd.xml etc.
# Clone of the official challenge repository, for the official scorer:
export MTAH_OFFICIAL_REPO=/path/to/more-than-a-hack-2027

python make_submission.py --model v2 --min-s 2 --fill --decoder hsmm --drop-outside
# -> out/submissions/v2_min2s_fill_hsmm_rule.csv
python "$MTAH_OFFICIAL_REPO/evaluation/validate_submission.py" out/submissions/v2_min2s_fill_hsmm_rule.csv
md5sum out/submissions/v2_min2s_fill_hsmm_rule.csv  # cf68da6c29f3bad45d58fdcada833a09
```

The run is deterministic (fixed seeds). Data locations can be overridden with `METRICA_SAMPLE_DIR`,
`IDSSE_DIR` and `SKILLCORNER_DIR`. The script prints aggregates only (counts, shares, scores),
because the challenge rules forbid sending the match data to hosted AI services.

## Layout

| path | what |
|---|---|
| `make_submission.py` | fits on the external games and writes the submission CSV |
| `mtah/` | the method: features, classifier, decoders, ball fill, kick-off, scoring wrapper |
| `external/` | loaders for the open datasets, answer keys, gap simulator, controlled-possession rule, data fetch |
| `experiments/` | one script per entry in `EXPERIMENTS.md` |

## Data and credits

- Metrica Sports sample data, github.com/metrica-sports/sample-data (commit `e706dd5`).
- IDSSE / DFL open dataset: Bassek, Rein, Weber, Memmert, "An integrated dataset of spatiotemporal
  and event data in elite soccer", Scientific Data 12:195 (2025), figshare
  10.6084/m9.figshare.28196177. CC BY 4.0, data (c) Deutsche Fussball Liga.
- SkillCorner open data, github.com/SkillCorner/opendata (MIT). Used only in experiments, to check
  the gap simulator against real ball gaps (`experiments/e11_*`, `e12_*`).
