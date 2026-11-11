"""Sequence model (small transformer) over the per-frame clues: E27. Needs PyTorch.

The per-frame classifier of v2 sees each frame alone, with time only through rolling averages.
This model reads a window of ~50 s of the same clues at 25 fps and predicts H/A/D for every
0.2 s step.

Model: linear layer per frame -> convolution with stride 5 (25 fps -> 5 steps a second) ->
3 transformer encoder layers (width 96, 4 heads) -> three outputs per step:
  main    P(H, A, D) under the controlled rule (the label we are scored on)
  touch   P(H, A, D) at touch level (the external label before the controlled rule)
  ball    where the hidden ball is, as its distance to each team's nearest player (only on
          steps where the simulator hid the ball: "restore what was deleted")
The last two are training signals only.

Team symmetry as in v2: each training window is seen as it is or team-swapped at random, and a
prediction is averaged with the prediction of the swapped input.

A "copy" is a dict with X (n, F) clues, names, y (controlled labels), gt (touch-level labels),
aux (true_distances), game and tag.
"""

import time

import numpy as np
import torch
from torch import nn

from mtah import features
from mtah.labels import X as UNGRADED

STRIDE = 5        # frames per step: 0.2 s, well inside the scorer's tolerance (EVALUATION.md)
WINDOW = 256      # steps per training window: 51 s, four mean possessions
WIDTH, LAYERS, HEADS = 96, 3, 4
BATCH, LR, DECAY, DROPOUT = 32, 1e-3, 0.05, 0.1
AUX_WEIGHT = 0.3
SWAP = np.array([1, 0, 2, 3])


def true_distances(clean, players, teams, pitch):
    """(n, 2) metres from the true ball to the nearest home / away player; NaN without a ball."""
    scale = np.asarray(pitch, dtype=float)
    return np.column_stack([features._nearest_m(players, teams, clean * scale, scale, t)
                            for t in (0, 1)]).astype(np.float32)


class Scaler:
    """Clip each clue to its 1st-99th percentile, standardise, NaN -> 0 plus a 'missing' flag.
    Fitted on the training copies and their team swaps, so it is the same for both teams."""

    FIELDS = ("lo", "hi", "mean", "std", "flag")

    def fit(self, train, cols):
        sample = np.concatenate([d["X"][::25, cols] for d in train])
        names = [train[0]["names"][i] for i in cols]
        sample = np.concatenate([sample, features.swap_teams(sample, names)])
        self.lo, self.hi = np.nanpercentile(sample, (1, 99), axis=0)
        clipped = np.clip(sample, self.lo, self.hi)
        self.mean, self.std = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0) + 1e-6
        self.flag = np.flatnonzero(np.isnan(sample).any(axis=0))
        return self

    def __call__(self, X):
        Z = (np.clip(X, self.lo, self.hi) - self.mean) / self.std
        missing = np.isnan(Z)
        return np.hstack([np.where(missing, 0.0, Z), missing[:, self.flag]]).astype(np.float32)


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, WIDTH), nn.GELU())
        self.down = nn.Conv1d(WIDTH, WIDTH, STRIDE, stride=STRIDE)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, 2 * WIDTH, DROPOUT, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(WIDTH)
        self.main, self.touch, self.ball = nn.Linear(WIDTH, 3), nn.Linear(WIDTH, 3), \
            nn.Linear(WIDTH, 2)
        pos = torch.arange(WINDOW)[:, None] / (10000 ** (torch.arange(0, WIDTH, 2) / WIDTH))
        self.register_buffer("pos", torch.cat([pos.sin(), pos.cos()], dim=1))

    def forward(self, x):  # (batch, WINDOW * STRIDE, n_in)
        h = self.down(self.inp(x).transpose(1, 2)).transpose(1, 2)
        h = self.norm(self.encoder(h + self.pos[:h.shape[1]]))
        return self.main(h), self.touch(h), self.ball(h)


def prepare(d, cols):
    """One copy for training: inputs at frame rate (unscaled), targets at the centre of each
    0.2 s step."""
    y, touch, aux = d["y"], d["gt"], d["aux"]
    n = len(y) // STRIDE * STRIDE
    mid = np.arange(STRIDE // 2, n, STRIDE)
    hidden = np.isnan(d["X"][mid, d["names"].index("d_home")])  # no ball in the damaged track
    target = np.log1p(aux[mid])
    target[~hidden] = np.nan
    X = d["X"][:n] if len(cols) == d["X"].shape[1] else d["X"][:n, cols]
    return dict(X=X, names=[d["names"][i] for i in cols], y=y[mid].astype(np.int64),
                touch=touch[mid].astype(np.int64), ball=target.astype(np.float32))


def windows(items, rng):
    """[(copy, first step, swap teams?)] covering every copy once from a random offset, each
    window seen as it is or team-swapped at random; windows with under a fifth of graded steps
    are dropped."""
    out = []
    for it in items:
        for a in range(int(rng.integers(0, WINDOW)), len(it["y"]) - WINDOW + 1, WINDOW):
            if np.mean(it["y"][a:a + WINDOW] != UNGRADED) >= 0.2:
                out.append((it, a, bool(rng.integers(0, 2))))
    return [out[j] for j in rng.permutation(len(out))]


def batch(pick, scaler):
    """Tensors (inputs, main labels, touch labels, hidden-ball targets) of a list of windows."""
    xs, ys, ts, bs = [], [], [], []
    for it, a, swap in pick:
        x = it["X"][a * STRIDE:(a + WINDOW) * STRIDE]
        y, t, b = (it[k][a:a + WINDOW] for k in ("y", "touch", "ball"))
        if swap:
            x, y, t, b = features.swap_teams(x, it["names"]), SWAP[y], SWAP[t], b[:, ::-1]
        xs.append(x), ys.append(y), ts.append(t), bs.append(b)
    x = scaler(np.concatenate(xs)).reshape(len(pick), WINDOW * STRIDE, -1)
    return [torch.from_numpy(np.ascontiguousarray(v))
            for v in (x, np.stack(ys), np.stack(ts), np.stack(bs))]


def train_net(items, scaler, epochs, aux, seed, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(items[0]["X"].shape[1] + len(scaler.flag))
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=DECAY)
    steps = epochs * (len(windows(items, np.random.default_rng(0))) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    ce = nn.CrossEntropyLoss(ignore_index=int(UNGRADED))
    done = 0
    for epoch in range(epochs):
        net.train()
        order, total, t0 = windows(items, rng), 0.0, time.time()
        for b in range(0, len(order) - BATCH + 1, BATCH):
            if done == steps:
                break
            x, y, y_touch, target = batch(order[b:b + BATCH], scaler)
            main, touch, ball = net(x)
            loss = ce(main.reshape(-1, 3), y.reshape(-1))
            if aux:
                loss = loss + AUX_WEIGHT * ce(touch.reshape(-1, 3), y_touch.reshape(-1))
                known = ~torch.isnan(target)
                if known.any():
                    loss = loss + AUX_WEIGHT * nn.functional.huber_loss(ball[known], target[known])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            done += 1
            total += float(loss)
        if epoch in (0, epochs - 1) or epoch % 3 == 2:
            log("    epoch {:>2}: loss {:.4f}, {:.0f} s".format(
                epoch + 1, total / max(1, len(order) // BATCH), time.time() - t0))
    return net.eval()


@torch.no_grad()
def predict(net, x):
    """Per-frame P(H, A, D) of one copy: overlapping windows, centre-weighted, then linear
    interpolation from steps to frames."""
    n_steps = len(x) // STRIDE
    logits = np.zeros((n_steps, 3))
    weight = np.zeros(n_steps)
    tri = 1.0 - np.abs(np.linspace(-1, 1, WINDOW)) * 0.9
    starts = list(range(0, max(1, n_steps - WINDOW + 1), WINDOW // 2))
    if starts[-1] != n_steps - WINDOW:
        starts.append(max(0, n_steps - WINDOW))
    for b in range(0, len(starts), 64):
        chunk = starts[b:b + 64]
        out = net(torch.from_numpy(np.stack([x[a * STRIDE:(a + WINDOW) * STRIDE]
                                             for a in chunk])))[0].numpy()
        for a, o in zip(chunk, out):
            logits[a:a + WINDOW] += o * tri[:, None]
            weight[a:a + WINDOW] += tri
    logits /= weight[:, None]
    frames = np.arange(len(x))
    mid = np.arange(STRIDE // 2, n_steps * STRIDE, STRIDE)
    z = np.column_stack([np.interp(frames, mid, logits[:, k]) for k in range(3)])
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def symmetric(net, X, names, scaler):
    """(n, 3) probabilities for the clue matrix X: the prediction averaged with its team swap."""
    p = predict(net, scaler(X))
    q = predict(net, scaler(features.swap_teams(X, names)))
    out = 0.5 * (p + q[:, [1, 0, 2]])
    if len(out) < len(X):  # the frames after the last whole step
        out = np.vstack([out, np.repeat(out[-1:], len(X) - len(out), axis=0)])
    return out


def fit_folds(train, cols, epochs, aux, log):
    """Three models, each trained without one third of the games (the folds of mtah.v2.fit).
    Returns ({game|tag: out-of-fold probabilities}, [(net, scaler)])."""
    names = [train[0]["names"][i] for i in cols]
    ids = sorted({d["game"] for d in train})
    oof, models = {}, []
    for k, fold in enumerate([ids[i::3] for i in range(3)]):
        inside = [d for d in train if d["game"] not in fold]
        scaler = Scaler().fit(inside, cols)
        log("  fold {}: {} copies, {} inputs".format(k + 1, len(inside),
                                                    len(cols) + len(scaler.flag)))
        net = train_net([prepare(d, cols) for d in inside], scaler, epochs, aux, seed=k, log=log)
        for d in train:
            if d["game"] in fold:
                oof["{}|{}".format(d["game"], d["tag"])] = symmetric(net, d["X"][:, cols], names,
                                                                    scaler)
        models.append((net, scaler))
    return oof, models


def ensemble(models, X, names):
    """Mean probabilities of the fold models."""
    return sum(symmetric(net, X, names, scaler) for net, scaler in models) / len(models)


def fuse(p, q):
    """Geometric mean of two probability arrays, renormalised: both models count the same."""
    z = 0.5 * (np.log(np.clip(p, 1e-6, 1)) + np.log(np.clip(q, 1e-6, 1)))
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def save(path, models, names, **extra):
    """Weights and scalers of the fold models, the clue names they expect, and extra scalars."""
    folds = [dict(state=net.state_dict(),
                  scaler={k: torch.as_tensor(getattr(scaler, k)) for k in Scaler.FIELDS})
             for net, scaler in models]
    torch.save(dict(names=list(names), folds=folds, **extra), path)


def load(path):
    """dict(names, models=[(net, scaler)], ...) from a file written by save()."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    models = []
    for fold in bundle.pop("folds"):
        scaler = Scaler()
        for k in Scaler.FIELDS:
            setattr(scaler, k, fold["scaler"][k].numpy())
        net = Net(len(bundle["names"]) + len(scaler.flag))
        net.load_state_dict(fold["state"])
        models.append((net.eval(), scaler))
    bundle["models"] = models
    return bundle
