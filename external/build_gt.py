"""Frame-level H/A/D answer keys for the Metrica sample games, derived from their event data.

    python external/build_gt.py [--out DIR]

Writes Sample_Game_<g>_gt.csv (+ _meta.json) in the competition's frame,label format, so the
official scorer can grade any of our predictions on these games with
`run_evaluation.py --gt <path>`.

Rules (Metrica event definitions, documentation/events-definitions.pdf):

Who has the ball while it is alive. Possession changes only when the other team performs an
on-ball event (RECOVERY, PASS, SET PIECE, ...). A loose or contested ball stays with the last
team that had it, so BALL LOST does not hand the ball over by itself and CHALLENGE never does.
This is also the competition's rule ("possession stays with the last team that had it").

When the ball is dead. A stoppage starts at
  - BALL OUT                                  end frame (the ball crosses the line)
  - SHOT whose result is GOAL or OUT          end frame
  - FAULT RECEIVED                            start frame (the whistle)
  - CHALLENGE with a FAULT subtype            end frame
  - BALL LOST / PASS with FORCED, END HALF or OFFSIDE   end frame
and lasts until the start frame of the next on-ball event (the restart: a SET PIECE, or a goal
kick, which Metrica logs as a PASS / BALL LOST / BALL OUT with subtype GOAL KICK). A SET PIECE
not preceded by any of the markers above (an untagged foul or handball) gets a stoppage from
the end of the last preceding event; these fallbacks are counted and reported.

Graded window per half: from the kick-off to END HALF (or the last event if there is none).
Everything else is X, which also keeps the two halves separate for the scorer.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica_sample as ms  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                           "external_data", "labels", "metrica_sample")
FPS = 25

POSSESSION_TYPES = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT", "CARRY",
                    "FAULT RECEIVED"}
RESTART_TYPES = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT", "CARRY"}
REFEREE_STOP = {"FORCED", "END HALF", "OFFSIDE"}

H, A, D, X = "H", "A", "D", "X"


def stoppage_frame(ev):
    """Frame (counted from 1) where this event kills the ball, or None."""
    tokens = set(ev.subtype.split("-")) if ev.subtype else set()
    if ev.type == "BALL OUT":
        return ev.end
    if ev.type == "SHOT" and tokens & {"GOAL", "OUT"}:
        return ev.end
    if ev.type == "FAULT RECEIVED":
        return ev.start
    if ev.type == "CHALLENGE" and "FAULT" in tokens:
        return ev.end
    if ev.type in ("BALL LOST", "PASS") and tokens & REFEREE_STOP:
        return ev.end
    return None


def half_windows(events):
    """[(period, first, last)] graded frames per half, counted from 1, inclusive."""
    out = []
    for period, ev in events.groupby("period"):
        kickoffs = ev[(ev.type == "SET PIECE") & ev.subtype.str.contains("KICK OFF")]
        first = int(kickoffs.start.min()) if len(kickoffs) else int(ev.start.min())
        end_half = ev[ev.subtype.str.contains("END HALF")]
        last = int(end_half.end.max()) if len(end_half) else int(ev.end.max())
        out.append((int(period), first, last))
    return out


def build_labels(events, n_frames):
    """(labels array of 'H'/'A'/'D'/'X', diagnostics dict)."""
    labels = np.full(n_frames, X, dtype="<U1")
    dead = np.zeros(n_frames, dtype=bool)
    windows = half_windows(events)
    n_marked, n_fallback = 0, 0

    for period, first, last in windows:
        ev = events[events.period == period]

        # Owner of a live ball: the team of the latest on-ball event, forward-filled.
        change = np.full(n_frames, -1, dtype=np.int8)
        for row in ev[ev.type.isin(POSSESSION_TYPES)].itertuples():
            change[row.start - 1] = row.team
        idx = np.where(change >= 0, np.arange(n_frames), -1)
        np.maximum.accumulate(idx, out=idx)
        owner = np.where(idx >= 0, change[np.maximum(idx, 0)], -1)

        restarts = ev[ev.type.isin(RESTART_TYPES)].start.to_numpy()
        for row in ev.itertuples():
            s = stoppage_frame(row)
            if s is None:
                continue
            later = restarts[restarts > s]
            r = int(later.min()) if len(later) else last + 1
            dead[s - 1:r - 1] = True
            n_marked += 1

        # SET PIECEs with no tagged stoppage before them.
        live_events = ev[ev.type != "CARD"]
        for row in ev[ev.type == "SET PIECE"].itertuples():
            r = row.start
            if r <= first or dead[r - 2]:
                continue
            before = live_events[live_events.start < r]
            if not len(before):
                continue
            s = int(before.end.max())
            dead[min(s, r - 1) - 1:r - 1] = True
            n_fallback += 1

        sl = slice(first - 1, last)
        window_owner = owner[sl]
        labels[sl] = np.where(dead[sl], D, np.where(window_owner == ms.HOME, H,
                                                    np.where(window_owner == ms.AWAY, A, X)))

    return labels, {"windows": windows, "stoppages_tagged": n_marked,
                    "stoppages_fallback": n_fallback}


def runs(labels):
    """[(label, start, stop)] maximal runs, stop exclusive."""
    br = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], br))
    stops = np.concatenate((br, [len(labels)]))
    return [(labels[a], int(a), int(b)) for a, b in zip(starts, stops)]


def structure(labels):
    graded = labels != X
    spells = [(c, b - a) for c, a, b in runs(labels) if c in (H, A)]
    dead = [(b - a) for c, a, b in runs(labels) if c == D]
    minutes = graded.sum() / FPS / 60
    h = sum(n for c, n in spells if c == H)
    return {
        "graded_frames": int(graded.sum()),
        "graded_minutes": round(float(minutes), 1),
        "dead_share": round(float((labels == D).sum() / graded.sum()), 3),
        "possessions": len(spells),
        "possessions_per_min": round(len(spells) / minutes, 2),
        "possession_mean_s": round(float(np.mean([n for _, n in spells])) / FPS, 1),
        "possession_median_s": round(float(np.median([n for _, n in spells])) / FPS, 1),
        "possessions_under_2s": int(sum(n < 2 * FPS for _, n in spells)),
        "home_share": round(h / sum(n for _, n in spells), 3),
        "dead_intervals": len(dead),
        "dead_median_s": round(float(np.median(dead)) / FPS, 1),
        "dead_over_120s": int(sum(n > 120 * FPS for n in dead)),
        "unlabelled_live_frames": int(((labels == X) & graded).sum()),
    }


def tracking_checks(labels, ball, players, teams):
    """Independent sanity checks against the tracking (which the labels never looked at)."""
    n = min(len(labels), len(ball))
    labels, ball, players, teams = labels[:n], ball[:n], players[:n], teams[:n]
    seen = ~np.isnan(ball).any(axis=1)
    outside = seen & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    live = np.isin(labels, (H, A))
    dead = labels == D

    scale = np.array(ms.PITCH)
    dist = np.linalg.norm((players - ball[:, None, :]) * scale, axis=2)
    dist[np.isnan(dist) | (teams < 0)] = np.inf
    nearest = np.argmin(dist, axis=1)
    near_team = teams[np.arange(n), nearest]
    ok = live & seen & np.isfinite(dist.min(axis=1))
    agree = (near_team[ok] == np.where(labels[ok] == H, ms.HOME, ms.AWAY)).mean()
    return {
        "ball_seen_live": round(float(seen[live].mean()), 3),
        "ball_seen_dead": round(float(seen[dead].mean()), 3),
        "ball_outside_pitch_live": round(float(outside[live].mean()), 3),
        "ball_outside_pitch_dead": round(float(outside[dead].mean()), 3),
        "nearest_player_agrees_live": round(float(agree), 3),
    }


def write(out_dir, game, labels, meta):
    stem = os.path.join(out_dir, "Sample_Game_{}_gt".format(game))
    with open(stem + ".csv", "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(labels))
    with open(stem + "_meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    return stem + ".csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for game in ms.GAMES:
        events = ms.load_events(game)
        ball, players, teams = ms.load_tracking(game)
        labels, info = build_labels(events, len(ball))
        stats = structure(labels)
        checks = tracking_checks(labels, ball, players, teams)
        meta = {"game": "Metrica Sample_Game_{}".format(game), "n_video_frames": len(labels),
                "fps": FPS, "source": "github.com/metrica-sports/sample-data (events)",
                "graded_windows_frames_from_1": info["windows"],
                "stoppages_tagged": info["stoppages_tagged"],
                "stoppages_fallback": info["stoppages_fallback"],
                "structure": stats, "tracking_checks": checks}
        path = write(args.out, game, labels, meta)
        print("== Game {}  -> {}".format(game, os.path.relpath(path)))
        print("   windows {}  stoppages tagged {}  fallback {}".format(
            info["windows"], info["stoppages_tagged"], info["stoppages_fallback"]))
        print("   " + "  ".join("{}={}".format(k, v) for k, v in stats.items()))
        print("   " + "  ".join("{}={}".format(k, v) for k, v in checks.items()))


if __name__ == "__main__":
    main()
