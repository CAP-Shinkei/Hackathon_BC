"""E04: kick-off detector on the external games, against the event-data kick-offs.

The search window imitates a Smart Tagging phase: a few seconds around the true kick-off, placed
at random offsets. Run on clean and on gap-damaged ball tracks.

    python experiments/e04_kickoff_external.py
"""

import numpy as np

from common import external_games
import metrica_sample as ms  # noqa: E402  (path set by common)
from gapsim import Params, damage
from mtah.kickoff import detect


def main():
    rng = np.random.default_rng(0)
    errors = {"clean": [], "live35": [], "live50": []}
    misses = {k: 0 for k in errors}
    for g, game in zip(external_games(), ms.GAMES):
        ev = ms.load_events(game)
        kicks = ev[(ev.type == "SET PIECE") & ev.subtype.str.contains("KICK OFF")].start.to_numpy() - 1
        balls = {"clean": g["ball"]}
        for name, r in (("live35", 0.35), ("live50", 0.5)):
            balls[name] = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                 Params(live_missing=r), seed=3)[0]
        for k in kicks:
            for _ in range(5):
                lo = int(k - rng.integers(25, 150))
                hi = int(k + rng.integers(25, 150))
                for name, ball in balls.items():
                    f = detect(ball, g["pitch"], lo, hi)
                    if f is None:
                        misses[name] += 1
                    else:
                        errors[name].append(f - k)
    for name, e in errors.items():
        e = np.abs(np.array(e))
        print("{:<7} detected {:>3}/{:<3} |error| median {:.0f} frames, 90th pct {:.0f}, max {:.0f}"
              .format(name, len(e), len(e) + misses[name], np.median(e), np.percentile(e, 90),
                      e.max()))


if __name__ == "__main__":
    main()
