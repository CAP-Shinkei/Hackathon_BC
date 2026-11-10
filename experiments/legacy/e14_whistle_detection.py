"""E14 whistle detector: reproducible validation on EXTERNAL audio only.

Sources (Wikimedia Commons, downloaded and converted locally with ffmpeg):
- reference blast: "218318_splicesound_referee-whistle-blow-gymnasium.wav"
    ffmpeg -i <src> -ac 1 -ar 16000 -acodec pcm_s16le whistle16k.wav
- stadium crowd:  "WWS FootballAustriavs.Sweden.ogg"
    ffmpeg -i <src> -ac 1 -ar 16000 -acodec pcm_s16le crowd16k.wav

Method: mix the reference blast into the crowd at several positions and SNR levels
(power ratio, dB), run the shipped detector (fixed constants, nothing fitted here),
and report recall per SNR plus false alarms on 227 s of pure crowd.

Usage: python experiments/e14_whistle_detection.py WHISTLE16K.wav CROWD16K.wav
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from mtah.whistle import detect_events, read_wav

POSITIONS_S = [10.0, 30.0, 60.0, 100.0, 150.0]
SNRS_DB = [10, 5, 0, -5]


def mix_at(crowd, blast, sr, t0, snr_db):
    n = len(blast)
    i0 = int(t0 * sr)
    seg = crowd[i0:i0 + n]
    p_w = float((blast ** 2).mean())
    p_c = float((seg ** 2).mean()) + 1e-12
    g = np.sqrt(p_c * 10 ** (snr_db / 10) / (p_w + 1e-12))
    out = crowd.copy()
    out[i0:i0 + n] += g * blast
    return out


def main():
    blast, sr1 = read_wav(sys.argv[1])
    crowd, sr2 = read_wav(sys.argv[2])
    assert sr1 == sr2 == 16000, "expected 16 kHz mono wav (see docstring)"
    dur = len(blast) / sr1
    print(f"reference blast {dur:.2f} s, crowd {len(crowd) / sr2:.0f} s")

    for snr in SNRS_DB:
        hits = 0
        for t0 in POSITIONS_S:
            events = detect_events(mix_at(crowd, blast, sr1, t0, snr), sr1)
            if any(s < t0 + dur and e > t0 for s, e in events):
                hits += 1
        print(f"SNR {snr:+d} dB: {hits}/{len(POSITIONS_S)} blasts detected")

    fa = detect_events(crowd, sr2)
    print(f"pure crowd: {len(fa)} false alarms in {len(crowd) / sr2:.0f} s")


if __name__ == "__main__":
    main()
