"""A trained possession bundle: fit on damaged games, predict a match, write the submission CSV.

Prediction is the experiment chain without gap simulation: drop far-outside ball positions,
featurise, average the fitted models' probabilities, then decode each live block with the
duration and transition priors counted on the training labels.
"""
import importlib
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np

from mtah import decode, gapsim, model, rules
from mtah.schema import A, D, H, LETTERS, X

HEADER = "frame,label"
NEWLINE = "\n"
WRITABLE_CODES = (H, A, D)  # X is an abstention and has no letter in the CSV
FEATURE_PACKAGE = "mtah"


@dataclass(frozen=True)
class Bundle:
    models: tuple
    names: tuple
    dur_logpmf: np.ndarray
    trans_log: np.ndarray
    temperature: float
    step: int
    min_frames: int
    drop_outside: bool
    feature_module: str


def _feature_module(name):
    try:
        return importlib.import_module(FEATURE_PACKAGE + "." + name)
    except ImportError as error:
        raise ValueError("unknown feature module {!r}: {}".format(name, error)) from error


def _check_training_inputs(matches, train_regimes, model_seeds):
    if not matches:
        raise ValueError("no training matches given")
    if any(match.labels is None for match in matches):
        raise ValueError("every training match needs labels")
    if not train_regimes:
        raise ValueError("at least one training regime is required")
    if not model_seeds:
        raise ValueError("at least one model seed is required")


def _featurise_damaged(match, regime_name, seed, drop_outside, module):
    damaged = gapsim.damage(match, gapsim.REGIMES[regime_name], seed)
    if drop_outside:
        damaged = rules.drop_far_outside(damaged)
    return module.featurise(damaged)


def _featurise_as_loaded(match, drop_outside, module):
    return module.featurise(rules.drop_far_outside(match) if drop_outside else match)


def _training_features(matches, real_matches, train_regimes, train_seed, drop_outside, module):
    """Same order as mtah.experiment.fit_fold: damaged clean games first, then real-gap games."""
    X_list, y_list, blocks_list, names = [], [], [], None
    for match in matches:
        for k, regime in enumerate(train_regimes):
            X, names = _featurise_damaged(match, regime, train_seed + k, drop_outside, module)
            X_list.append(X)
            y_list.append(match.labels)
            blocks_list.append(match.blocks)
    for match in real_matches:
        X, names = _featurise_as_loaded(match, drop_outside, module)
        X_list.append(X)
        y_list.append(match.labels)
        blocks_list.append(match.blocks)
    return X_list, y_list, blocks_list, names


def _fit_models(X_list, y_list, blocks_list, swap, params, model_seeds):
    return tuple(
        model.fit(X_list, y_list, blocks_list, swap,
                  model.Params(**{**asdict(params), "seed": seed}))
        for seed in model_seeds)


def train_bundle(matches, train_regimes, train_seed, model_seeds, params, temperature, step,
                 min_frames, drop_outside=True, feature_module="features", real_matches=()):
    """Fit one model per seed on every match under every training regime, with priors from labels.

    `real_matches` are games whose tracking already has genuine gaps; they are used as loaded.
    """
    real_matches = tuple(real_matches)
    _check_training_inputs(tuple(matches) + real_matches, train_regimes, model_seeds)
    module = _feature_module(feature_module)
    X_list, y_list, blocks_list, names = _training_features(
        matches, real_matches, train_regimes, train_seed, drop_outside, module)
    swap = module.swap_index(names)
    fitted = _fit_models(X_list, y_list, blocks_list, swap, params, model_seeds)
    labels = [match.labels for match in tuple(matches) + real_matches]
    blocks = [match.blocks for match in tuple(matches) + real_matches]
    return Bundle(
        models=fitted,
        names=tuple(names),
        dur_logpmf=decode.count_durations(labels, blocks, step=step),
        trans_log=decode.count_transitions(labels, blocks),
        temperature=float(temperature),
        step=int(step),
        min_frames=int(min_frames),
        drop_outside=bool(drop_outside),
        feature_module=feature_module,
    )


def save_bundle(bundle, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, Bundle):
        raise ValueError("{} does not hold a possession Bundle".format(path))
    return bundle


def _kicked_off_blocks(blocks, kickoff_frames):
    """Blocks cut to start at their kick-off; a block whose kick-off is past its end is dropped."""
    if kickoff_frames is None:
        return tuple(blocks)
    if len(kickoff_frames) != len(blocks):
        raise ValueError("kickoff: {} frames given for {} blocks".format(
            len(kickoff_frames), len(blocks)))
    cut = []
    for (start, stop), kickoff in zip(blocks, kickoff_frames):
        begin = max(start, int(kickoff))
        if begin < stop:
            cut.append((begin, stop))
    return tuple(cut)


def _check_feature_names(names, expected):
    if tuple(names) != tuple(expected):
        raise ValueError("feature names differ from the bundle ({} vs {} columns)".format(
            len(names), len(expected)))


def _mean_proba(bundle, X, swap):
    return np.mean([model.predict_proba(fitted, X, swap) for fitted in bundle.models], axis=0)


def predict_codes(bundle, match, kickoff_frames=None):
    """int8 codes [T] for a match. Frames outside the decoded blocks, and before a kick-off, are D."""
    prepared = rules.drop_far_outside(match) if bundle.drop_outside else match
    module = _feature_module(bundle.feature_module)
    X, names = module.featurise(prepared)
    _check_feature_names(names, bundle.names)
    proba = _mean_proba(bundle, X, module.swap_index(names))
    blocks = _kicked_off_blocks(match.blocks, kickoff_frames)
    return decode.decode_blocks(proba, blocks, bundle.dur_logpmf, bundle.trans_log,
                                temperature=bundle.temperature, step=bundle.step,
                                min_frames=bundle.min_frames)


def _csv_text(codes):
    values = np.asarray(codes)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("codes must be a non-empty 1-D array")
    abstain = int(np.count_nonzero(values == X))
    if abstain or not np.isin(values, WRITABLE_CODES).all():
        raise ValueError("only H, A and D can be written; X (abstain) is never written "
                         "({} abstentions)".format(abstain))
    rows = "".join("{},{}{}".format(index, LETTERS[code], NEWLINE)
                   for index, code in enumerate(values.astype(np.intp)))
    return HEADER + NEWLINE + rows


def write_csv(codes, path):
    """Write the frame,label CSV, one row per frame; byte-identical for identical codes."""
    text = _csv_text(codes)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="") as handle:
        handle.write(text)
    return path
