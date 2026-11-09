"""Label codes and the frame,label CSV format shared by submissions and answer keys.

Codes are the official scorer's (evaluation/possession_metrics.py): H=0, A=1, D=2, X=3.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])


def read_labels(path):
    """Codes array from a frame,label CSV. Rows must be frames 0..n-1 in order."""
    with open(path) as handle:
        header = handle.readline().strip().lower()
        if header != "frame,label":
            raise ValueError("{}: header is {!r}".format(path, header))
        letters = [line.rstrip("\r\n").split(",")[1].strip().upper() for line in handle]
    lookup = {name: code for code, name in enumerate(LETTERS)}
    return np.array([lookup[c] for c in letters], dtype=np.int8)


def write_labels(path, codes):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[codes]))
    return path


def runs(codes):
    """[(code, start, stop)] maximal runs, stop exclusive."""
    codes = np.asarray(codes)
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], br))
    stops = np.concatenate((br, [len(codes)]))
    return [(int(codes[a]), int(a), int(b)) for a, b in zip(starts, stops)]


def graded_blocks(gt_codes):
    """[(start, stop)] spans where the answer key is not X (the halves), stop exclusive."""
    return [(a, b) for c, a, b in runs(np.asarray(gt_codes) != X) if c]
