"""Experimental referee-whistle event exporter (standalone; NOT wired into the submission).

What this is: a local detector that turns an extracted audio track into a CSV of candidate
whistle-blast time intervals. What it is NOT: a dead-ball signal usable for cutting possessions.
A whistle both STOPS play and AUTHORISES restarts (kick-offs, and any free kick the referee
requires to be whistled - IFAB Law 9 works in both directions), so a detected blast does not by
itself establish that the interval is dead ball. Classifying each blast (stop vs restart) would
need possession-level evaluation on external games with audio, which no available external
dataset provides; the rules also forbid calibrating or checking anything against the match
video/output. So the integration stays out of the submission pipeline on purpose.

The match audio, if ever used, is an INFERENCE INPUT ONLY: band and threshold are fixed here
from EXTERNAL audio, never adapted to the match (the clarified rules forbid training, fitting
or calibrating on the provided video). Audio is extracted locally, nothing is uploaded:

    ffmpeg -i "$MTAH_DATA_DIR/tactical/BAR-ATH_tactical.mp4" -ac 1 -ar 16000 \
        -acodec pcm_s16le tactical_audio16k.wav
    python -m mtah.whistle tactical_audio16k.wav out/whistle_events.csv

Caveat on timing: the CSV times assume the audio starts at video t=0 (straight ffmpeg
extraction) and map to frames at 25 fps. Verifying that offset against the match video would be
calibration on the target and is not permitted, so event timing is unverified on the match.

Constants and their sources (none measured on the competition match):
- BAND_HZ = (2100, 6400): dominant band learned from a reference referee-whistle recording
  (Wikimedia Commons, "218318_splicesound_referee-whistle-blow-gymnasium.wav"): measured
  2312-5859 Hz; widened slightly to cover whistle-to-whistle variation.
- THRESH = 0.5: on external stadium audio (Wikimedia Commons, "WWS FootballAustriavs.Sweden.ogg")
  background windows score ~0.10 and true whistle windows ~0.87-0.99 down to -5 dB SNR, so any
  threshold in that wide gap separates them; 0.5 is the midpoint. Chosen on EXTERNAL audio only.
  (The 31-10 prototype used an adaptive per-recording quantile; that would be a statistic of the
  match audio itself, so it was replaced by this fixed value.)
- MIN_HOPS = 8 (0.256 s): a referee's whistle is a sustained tone; speech bursts and knocks are
  shorter. From the same external validation.
"""

import sys
import wave

import numpy as np

BAND_HZ = (2100.0, 6400.0)
THRESH = 0.5
WIN = 1024            # 64 ms at 16 kHz
HOP = 512             # 32 ms hop
MIN_HOPS = 8          # >= 0.25 s sustained
MERGE_GAP_S = 0.4     # one blast can dip briefly; keep it one event
FPS = 25


def read_wav(path):
    """Mono float signal and sample rate from a PCM wav (mixes down channels)."""
    with wave.open(path) as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("expected 16-bit PCM wav (extract with -acodec pcm_s16le)")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def whistle_score(x, sr):
    """Per-window band-energy ratio (0..1) and the hop in samples."""
    freqs = np.fft.rfftfreq(WIN, 1.0 / sr)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    nf = (len(x) - WIN) // HOP + 1
    if nf <= 0:
        return np.empty(0), HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(WIN), axis=1)) ** 2
    return spec[:, band].sum(axis=1) / (spec[:, 1:].sum(axis=1) + 1e-12), HOP


def detect_events(x, sr):
    """[(t_start_s, t_end_s)] whistle candidates: score > THRESH sustained >= MIN_HOPS hops,
    blasts closer than MERGE_GAP_S merged."""
    score, hop = whistle_score(x, sr)
    if len(score) == 0:
        return []
    above = score > THRESH
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.view(np.int8), [0]))))
    events = [[a * hop / sr, b * hop / sr]
              for a, b in zip(edges[0::2], edges[1::2]) if (b - a) >= MIN_HOPS]
    merged = []
    for ev in events:
        if merged and ev[0] - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = ev[1]
        else:
            merged.append(list(ev))
    return [(a, b) for a, b in merged]


def events_to_frames(events, n_frames, fps=FPS, pre_s=0.0, post_s=0.0):
    """[(start_frame, end_frame)] of dead-ball intervals, clipped to [0, n_frames), merged.

    pre_s/post_s pad each whistle: the dead ball starts at the whistle (the foul precedes it by
    an instant) and play resumes after; padding is a team decision, default 0 (the restart is
    left to the decoder / tagging boundaries)."""
    out = []
    for a, b in events:
        s = max(0, int(round((a - pre_s) * fps)))
        e = min(n_frames, int(round((b + post_s) * fps)))
        if e > s:
            out.append((s, e))
    out.sort()
    merged = []
    for s, e in out:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 1
    wav_path, out_csv = argv[1], argv[2]
    x, sr = read_wav(wav_path)
    events = detect_events(x, sr)
    frames = events_to_frames(events, n_frames=10**12)  # unclipped; seconds are exact
    with open(out_csv, "w") as fh:
        fh.write("start_frame,end_frame,start_s,end_s\n")
        for (s, e), (a, b) in zip(frames, events):
            fh.write("{},{},{:.2f},{:.2f}\n".format(s, e, a, b))
    # aggregates only: never print raw match-derived rows beyond the events file itself
    print("whistle events: {} ({} s of audio at {} Hz)".format(len(events), int(len(x) / sr), sr))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
