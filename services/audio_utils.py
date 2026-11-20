import os
import re
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import pyloudnorm as pyln
from config import SAMPLE_RATE, TARGET_LUFS, TRUE_PEAK_CEILING_DB, TEMP_DIR, REFERENCES_DIR


def ensure_temp_dir() -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def ensure_references_dir() -> str:
    os.makedirs(REFERENCES_DIR, exist_ok=True)
    return REFERENCES_DIR


def cleanup_files(paths: list[str]) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def extract_track_name(filename: str) -> str:
    base = os.path.basename(filename or "")
    name, _ = os.path.splitext(base)
    prefixes = (
        "gravação_",
        "gravacao_",
        "pre-mix+_",
        "pre-mix_",
        "premix_",
        "mix_",
        "ref_",
    )
    lower_name = name.lower()
    for p in prefixes:
        if lower_name.startswith(p):
            name = name[len(p):]
            break

    clean = re.sub(r"[^a-zA-Z0-9À-ÿ_-]", "_", name)
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_") or "faixa"


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE, mono: bool = False):
    data, sr = sf.read(file_path, dtype="float32", always_2d=True)
    if mono and data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)
    elif not mono and data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)

    if sr != target_sr:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down, axis=0).astype(np.float32)

    if mono:
        return data[:, 0], target_sr
    return data, target_sr


def save_wav(file_path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    peak = np.max(np.abs(audio)) if audio.size > 0 else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(file_path, audio.astype(np.float32), sr, subtype="PCM_24")
    return file_path


def normalize_lufs(audio: np.ndarray, sr: int = SAMPLE_RATE, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if np.isfinite(loudness) and loudness > -70.0:
        audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    ceiling_lin = 10.0 ** (TRUE_PEAK_CEILING_DB / 20.0)
    peak = np.max(np.abs(audio))
    if peak > ceiling_lin:
        audio = audio * (ceiling_lin / peak)
    return audio.astype(np.float32)
