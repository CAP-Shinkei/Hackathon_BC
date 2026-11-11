"""E25, our half of scoring the team's pipeline (CAP-Shinkei/Hackathon_BC `main`) on our test
sets: write the external games and E25's damaged test copies as .npz files that
e25_team_side.py reads from inside a checkout of their repository. Their package is also called
`mtah`, so the two halves cannot share a process.

    python experiments/e25_team_export.py <folder>

External open data only.
"""

import os
import sys

import numpy as np

from common import dfl_games, external_games, skillcorner_games
from controlled import controlled
from e25_benchmark import MIN_S, SC_HELD_OUT, TESTS
from gapsim import Params, damage


def save(folder, name, ball, players, teams, pitch, y):
    np.savez(os.path.join(folder, name + ".npz"), ball=ball.astype(np.float32),
             players=players.astype(np.float32), teams=teams, pitch=np.asarray(pitch), y=y)


def main(folder):
    os.makedirs(folder, exist_ok=True)
    for dataset, games in (("metrica", external_games()), ("dfl", dfl_games())):
        for g in games:
            y = controlled(g["gt"], MIN_S)
            save(folder, "train__{}__{}".format(dataset, g["name"]), g["ball"], g["players"],
                 g["teams"], g["pitch"], y)
            for tag, fields, seed in TESTS:
                ball, players = damage(g["ball"], g["players"], g["teams"], g["gt"], g["pitch"],
                                       Params(**fields), seed=seed)
                save(folder, "test__{}__{}__{}".format(dataset, g["name"], tag), ball, players,
                     g["teams"], g["pitch"], y)
        print(dataset, "written", flush=True)
    for i, g in enumerate(skillcorner_games()):
        role = "test" if i < SC_HELD_OUT else "train"
        save(folder, "{}__skillcorner__{}{}".format(role, g["name"],
                                                    "__sc_real" if role == "test" else ""),
             g["ball"], g["players"], g["teams"], g["pitch"], controlled(g["gt"], MIN_S))
    print("skillcorner written", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
