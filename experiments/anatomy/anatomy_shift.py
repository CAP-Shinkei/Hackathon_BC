"""Exploratory, EXTERNAL games only: spellF1 if every predicted direct turnover (H<->A) is moved
k frames later. Shows whether the turnover timing error is a systematic lag."""
import os, sys
from collections import defaultdict
import numpy as np
repo, dump_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from mtah.scoring import pm
KS = (-10, 0, 5, 10, 15, 20, 30)
res = defaultdict(lambda: defaultdict(list))
for fn in sorted(os.listdir(dump_dir)):
    if not (fn.startswith("anatomy_") and fn.endswith(".npz")):
        continue
    z = np.load(os.path.join(dump_dir, fn))
    for key in [k[2:] for k in z.files if k.startswith("y|")]:
        y, p = z["y|" + key], z["pred|" + key]
        runs = pm.valid_runs(y)
        t = np.flatnonzero((p[1:] != p[:-1]) & (p[1:] < 2) & (p[:-1] < 2)) + 1
        for k in KS:
            q = p.copy()
            for b in t:
                if k > 0:
                    e = b + k
                    nxt = np.flatnonzero(p[b:e] != p[b])
                    e = b + (nxt[0] if len(nxt) else k)
                    q[b:e] = p[b - 1]
                elif k < 0:
                    s = b + k
                    prv = np.flatnonzero(p[s:b] != p[b - 1])
                    s = s + (prv[-1] + 1 if len(prv) else 0)
                    q[s:b] = p[b]
            res[(fn[8:-4], key.split("|")[1])][k].append(pm.spell_scores(y, q, runs)["mean_f1"])
print("{:<28}".format("cell") + "".join("{:>8}".format("k=%d" % k) for k in KS))
for c in sorted(res):
    print("{:<28}".format(" ".join(c)) + "".join("{:8.3f}".format(np.mean(res[c][k])) for k in KS))
