#!/usr/bin/env python3
"""Audio comparison utility for C vs Rust parity (SNR, mel-corr, max_abs, memload)."""
import argparse
import json
import sys
import wave
import numpy as np


def read_wav_f32(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2 or nch != 1:
        raise ValueError(f"{path}: expected mono s16le, got ch={nch} width={sw}")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return sr, samples


def compute_snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    if n == 0:
        return -999.0
    r = ref[:n]
    t = test[:n]
    sig = float(np.sum(r.astype(np.float64) ** 2))
    err = float(np.sum((r.astype(np.float64) - t.astype(np.float64)) ** 2))
    if err == 0.0:
        return 999.0
    if sig == 0.0:
        return -999.0
    return 10.0 * np.log10(sig / err)


def compute_mel_corr(ref: np.ndarray, test: np.ndarray, sr: int = 24000) -> float:
    import librosa
    n = min(len(ref), len(test))
    if n < 512:
        return 0.0
    m1 = librosa.feature.melspectrogram(y=ref[:n], sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    m2 = librosa.feature.melspectrogram(y=test[:n], sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    l1 = np.log1p(m1).ravel()
    l2 = np.log1p(m2).ravel()
    if np.std(l1) < 1e-9 or np.std(l2) < 1e-9:
        return 0.0
    return float(np.corrcoef(l1, l2)[0, 1])


def memload_worker(mb: int, stop_flag_path: str):
    import os
    import time
    size = (mb * 1024 * 1024) // 2
    a = bytearray(size)
    b = bytearray(size)
    mv_a = memoryview(a)
    mv_b = memoryview(b)
    i = 0
    while not os.path.exists(stop_flag_path):
        mv_b[:] = mv_a
        mv_a[i % size] = (i & 0xFF)
        i += 4096
        if i % (4096 * 64) == 0:
            time.sleep(0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c_cmp = sub.add_parser("compare")
    c_cmp.add_argument("ref_wav")
    c_cmp.add_argument("test_wav")
    c_cmp.add_argument("--with-mel", action="store_true")

    c_ml = sub.add_parser("memload")
    c_ml.add_argument("--mb", type=int, default=256)
    c_ml.add_argument("--stop-file", required=True)

    args = ap.parse_args()
    if args.cmd == "memload":
        memload_worker(args.mb, args.stop_file)
        return

    sr1, ref = read_wav_f32(args.ref_wav)
    sr2, tst = read_wav_f32(args.test_wav)
    n = min(len(ref), len(tst))
    max_abs = float(np.max(np.abs(ref[:n] - tst[:n]))) if n > 0 else 1.0
    snr = compute_snr_db(ref, tst)
    out = {
        "ref_samples": int(len(ref)),
        "test_samples": int(len(tst)),
        "ref_sr": int(sr1),
        "test_sr": int(sr2),
        "snr_db": round(snr, 3),
        "max_abs_diff": round(max_abs, 6),
        "exact_bytes": (len(ref) == len(tst) and np.array_equal(ref, tst)),
    }
    if args.with_mel:
        out["mel_corr"] = round(compute_mel_corr(ref, tst, sr1), 5)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
