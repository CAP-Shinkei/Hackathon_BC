"""Per-frame possession classifier: P(home), P(away), P(dead), symmetric in the two teams.

One gradient-boosted model is trained on every graded frame plus its team-swapped copy, and
prediction averages the model over a frame and over its swapped twin. The averaged output is
therefore exactly symmetric, whatever the model learned.
"""
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.schema import A, D, H

GRADED_CODES = (H, A, D)
N_CLASSES = 3
# Columns are ordered H, A, D. Swapping teams exchanges the H and A columns and keeps D.
TEAM_SWAP_COLUMNS = np.array([1, 0, 2])
MIN_PROBABILITY = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class Params:
    max_iter: int = 300
    learning_rate: float = 0.08
    max_leaf_nodes: int = 63
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    stride: int = 5  # use every stride-th graded frame for training
    seed: int = 0


def _inside_blocks(n_frames, blocks):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        if not 0 <= start <= stop <= n_frames:
            raise ValueError("block {} is outside {} frames".format((start, stop), n_frames))
        mask[start:stop] = True
    return mask


def training_rows(labels, blocks, stride):
    """Indices of graded frames (H, A or D) inside blocks, keeping every stride-th one."""
    if stride < 1:
        raise ValueError("stride must be >= 1, got {}".format(stride))
    labels = np.asarray(labels)
    graded = _inside_blocks(len(labels), blocks) & np.isin(labels, GRADED_CODES)
    return np.flatnonzero(graded)[::stride]


def _validate_swap(swap_idx, n_features):
    swap = np.asarray(swap_idx)
    expected = np.arange(n_features)
    if swap.ndim != 1 or swap.shape[0] != n_features or not np.array_equal(np.sort(swap), expected):
        raise ValueError("swap_idx must be a permutation of {} feature columns".format(n_features))
    return swap.astype(np.intp)


def _swap_team_labels(labels):
    return np.where(labels == H, A, np.where(labels == A, H, labels)).astype(np.int8)


def _match_training_part(X, labels, blocks, swap_idx, stride):
    rows = training_rows(labels, blocks, stride)
    features = X[rows]
    targets = labels[rows]
    return [features, features[:, swap_idx]], [targets, _swap_team_labels(targets)]


def _training_set(X_list, y_list, blocks_list, swap_idx, stride):
    parts_X, parts_y = [], []
    for X, labels, blocks in zip(X_list, y_list, blocks_list):
        X = np.asarray(X, dtype=np.float32)
        labels = np.asarray(labels)
        if X.ndim != 2 or len(labels) != len(X):
            raise ValueError("each match needs 2-D features and one label per frame")
        swap = _validate_swap(swap_idx, X.shape[1])
        features, targets = _match_training_part(X, labels, blocks, swap, stride)
        parts_X.extend(features)
        parts_y.extend(targets)
    if not parts_X:
        raise ValueError("no training matches given")
    y_train = np.concatenate(parts_y)
    if len(y_train) == 0:
        raise ValueError("no graded frames inside the blocks to train on")
    return np.concatenate(parts_X), y_train


def fit(X_list, y_list, blocks_list, swap_idx, params=Params()):
    """Train one classifier on the matches plus their team-swapped copies."""
    if not len(X_list) == len(y_list) == len(blocks_list):
        raise ValueError("X_list, y_list and blocks_list must have one entry per match")
    X_train, y_train = _training_set(X_list, y_list, blocks_list, swap_idx, params.stride)
    estimator = HistGradientBoostingClassifier(
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        l2_regularization=params.l2_regularization,
        early_stopping=False,
        class_weight=None,
        random_state=params.seed,
    )
    estimator.fit(X_train, y_train)
    return estimator


def _class_probabilities(model, X):
    """Model output scattered into fixed H, A, D columns, whatever subset of classes it saw."""
    proba = model.predict_proba(X)
    full = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    full[:, np.asarray(model.classes_, dtype=np.intp)] = proba
    return full


def predict_proba(model, X, swap_idx):
    """float32 [T, 3] columns H, A, D; symmetric average of the frame and its swapped twin."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2-D, got shape {}".format(X.shape))
    swap = _validate_swap(swap_idx, X.shape[1])
    direct = _class_probabilities(model, X)
    mirrored = _class_probabilities(model, X[:, swap])[:, TEAM_SWAP_COLUMNS]
    return ((direct + mirrored) / 2).astype(np.float32)


def save(model, path):
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)


def temperature(proba, t):
    """Softens or sharpens probabilities: p ** (1/t), renormalised per row (computed in log space)."""
    if not t > 0:
        raise ValueError("temperature must be > 0, got {}".format(t))
    probs = np.asarray(proba, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise ValueError("proba must have shape [T, {}]".format(N_CLASSES))
    logits = np.log(np.maximum(probs, MIN_PROBABILITY)) / t
    scaled = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (scaled / scaled.sum(axis=1, keepdims=True)).astype(np.float32)
