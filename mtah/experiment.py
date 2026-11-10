"""Leave-one-game-out protocol on external games. Every experiment is a Config run through here.

A fold trains on all games but one, each damaged by the training regimes, and predicts the
held-out game under each evaluation regime and seed. Scores come from the official scorer only.
Features and fold probabilities are cached on disk because they dominate the cost.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from mtah import decode, features, gapsim, model, rules
from mtah.scoring import score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "out", "cache")


@dataclass(frozen=True)
class Config:
    name: str
    train_regimes: tuple = ("random35", "random50")
    eval_regimes: tuple = ("random35", "random50")
    eval_seeds: tuple = (7, 8)
    train_seed: int = 1
    model_seeds: tuple = (0,)
    params: model.Params = field(default_factory=model.Params)
    temperature: float = 1.0
    step: int = 5
    min_frames: int = 0
    drop_outside: bool = True
    feature_version: str = "v1"

    def model_key(self):
        """Everything that changes the fitted probabilities; decoder settings are left out."""
        keep = {k: v for k, v in asdict(self).items()
                if k in ("train_regimes", "train_seed", "model_seeds", "params",
                         "drop_outside", "feature_version")}
        return hashlib.md5(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def featurise_cached(match, regime_name, seed, cfg, featurise=features.featurise):
    """(X, names) of one damaged game. `regime_name` None means the game as loaded."""
    tag = "{}_{}_{}_{}_{}".format(match.game_id, regime_name, seed, cfg.feature_version,
                                  int(cfg.drop_outside))
    path = os.path.join(CACHE, "features", tag + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as data:
            return data["X"], [str(n) for n in data["names"]]
    damaged = match if regime_name is None else gapsim.damage(
        match, gapsim.REGIMES[regime_name], seed)
    if cfg.drop_outside:
        damaged = rules.drop_far_outside(damaged)
    X, names = featurise(damaged)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, names=np.array(names))
    return X, names


def fit_fold(train_matches, cfg, featurise=features.featurise):
    """One model per seed, trained on every training game under every training regime."""
    X_list, y_list, blocks_list, names = [], [], [], None
    for match in train_matches:
        for k, regime in enumerate(cfg.train_regimes):
            X, names = featurise_cached(match, regime, cfg.train_seed + k, cfg, featurise)
            X_list.append(X)
            y_list.append(match.labels)
            blocks_list.append(match.blocks)
    swap = features.swap_index(names)
    models = [model.fit(X_list, y_list, blocks_list, swap,
                        model.Params(**{**asdict(cfg.params), "seed": seed}))
              for seed in cfg.model_seeds]
    return models, swap


def fold_proba(held_out, train_matches, cfg, featurise=features.featurise):
    """{(regime, seed): P[T,3]} for the held-out game, cached per model key."""
    folder = os.path.join(CACHE, "proba", cfg.model_key())
    wanted = [(r, s) for r in cfg.eval_regimes for s in cfg.eval_seeds]
    paths = {key: os.path.join(folder, "{}_{}_{}.npy".format(held_out.game_id, *key))
             for key in wanted}
    if all(os.path.exists(p) for p in paths.values()):
        return {key: np.load(p) for key, p in paths.items()}
    models, swap = fit_fold(train_matches, cfg, featurise)
    os.makedirs(folder, exist_ok=True)
    out = {}
    for key in wanted:
        X, _ = featurise_cached(held_out, key[0], key[1], cfg, featurise)
        out[key] = np.mean([model.predict_proba(m, X, swap) for m in models], axis=0)
        np.save(paths[key], out[key])
    return out


def decode_and_score(proba, held_out, train_matches, cfg):
    labels = [m.labels for m in train_matches]
    blocks = [m.blocks for m in train_matches]
    codes = decode.decode_blocks(
        proba, held_out.blocks, decode.count_durations(labels, blocks, step=cfg.step),
        decode.count_transitions(labels, blocks), temperature=cfg.temperature,
        step=cfg.step, min_frames=cfg.min_frames)
    return score(held_out.labels, codes, blocks=held_out.blocks)


def run(matches, cfg, featurise=features.featurise, log=print):
    """Rows of {game, regime, seed, spell_f1, macro_f1, ...}, one per held-out evaluation."""
    rows = []
    for held_out in matches:
        train = [m for m in matches if m.game_id != held_out.game_id]
        for (regime, seed), proba in fold_proba(held_out, train, cfg, featurise).items():
            result = decode_and_score(proba, held_out, train, cfg)
            rows.append({"game": held_out.game_id, "regime": regime, "seed": seed, **result})
        done = [r for r in rows if r["game"] == held_out.game_id]
        log("{:<16} {:<14} spellF1 {:.3f}  macroF1 {:.3f}".format(
            cfg.name, held_out.game_id, np.mean([r["spell_f1"] for r in done]),
            np.mean([r["macro_f1"] for r in done])))
    return rows


def summarise(rows):
    """Mean over games of each game's mean, so a long game does not weigh more."""
    games = sorted({r["game"] for r in rows})
    per_game = {g: {k: float(np.mean([r[k] for r in rows if r["game"] == g]))
                    for k in ("spell_f1", "macro_f1", "d_f1")} for g in games}
    return {"per_game": per_game,
            **{k: float(np.mean([per_game[g][k] for g in games]))
               for k in ("spell_f1", "macro_f1", "d_f1")}}


def compare(base_rows, new_rows):
    """Paired per-game difference in spellF1: (mean gain, games improved, games total)."""
    base, new = summarise(base_rows)["per_game"], summarise(new_rows)["per_game"]
    gains = [new[g]["spell_f1"] - base[g]["spell_f1"] for g in base if g in new]
    return float(np.mean(gains)), int(np.sum(np.array(gains) > 0)), len(gains)
