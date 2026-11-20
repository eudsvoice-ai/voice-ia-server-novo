import os
import re
import numpy as np
import noisereduce as nr
import pyworld as pw
from scipy.signal import butter, sosfilt, lfilter
from pedalboard import (
    Pedalboard,
    NoiseGate,
    HighpassFilter,
    LowShelfFilter,
    PeakFilter,
    HighShelfFilter,
    Compressor,
    Reverb,
    Limiter,
)
from config import (
    SAMPLE_RATE,
    VALID_MATRICES,
    VOICE_PITCH_STRENGTH,
    VOICE_DEADZONE_CENTS,
    VOICE_ATTACK_MS,
)
from services.audio_utils import load_audio, save_wav, sanitize_filename, ensure_temp_dir

CHUNK_SECONDS = 30.0
CROSSFADE_MS = 100.0


def validate_not_already_treated(filename: str) -> None:
    base = os.path.basename(filename or "")
    if base.upper().startswith("IATRAT_"):
        raise ValueError("Faixa já foi tratada (prefixo iATRAT_ detectado). Não é permitido retratar.")


def build_clean_treated_filename(original_filename: str, matrix: str) -> str:
    clean_base = sanitize_filename(original_filename)
    tokens = [t for t in clean_base.split("_") if t]
    filtered = [
        t for t in tokens
        if t.lower() not in ("iatrat", "voz", "nylon", "aco", "aço")
    ]
    suffix = "_".join(filtered) if filtered else "faixa"
    return f"iATRAT_{matrix}_{suffix}.wav"


def _reduce_cellphone_noise(y: np.ndarray, sr: int, prop_decrease: float = 0.55) -> np.ndarray:
    if len(y) < sr * 0.5:
        return y
    return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=prop_decrease).astype(np.float32)


def _deesser_soft(y: np.ndarray, sr: int, threshold: float = 0.08, ratio: float = 0.45) -> np.ndarray:
    sos = butter(4, [5500, 9500], btype="bandpass", fs=sr, output="sos")
    sibilance = sosfilt(sos, y)
    env = np.abs(sibilance)
    b, a = butter(1, 40.0 / (sr / 2.0), btype="low")
    smooth_env = lfilter(b, a, env)
    gain = np.where(
        smooth_env > threshold,
        1.0 - ratio * np.clip((smooth_env - threshold) / (threshold + 1e-6), 0.0, 1.0),
        1.0,
    )
    return (y - sibilance * (1.0 - gain)).astype(np.float32)


def _pitch_correct_segment(segment: np.ndarray, sr: int) -> np.ndarray:
    y64 = segment.astype(np.float64)
    f0, t = pw.harvest(y64, sr, f0_floor=75.0, f0_ceil=650.0, frame_period=5.0)
    f0 = pw.stonemask(y64, f0, t, sr)
    sp = pw.cheaptrick(y64, f0, t, sr)
    ap = pw.d4c(y64, f0, t, sr)

    voiced = f0 > 0
    if not np.any(voiced):
        return segment.astype(np.float32)

    midi = np.zeros_like(f0)
    midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)
    nearest_midi = np.round(midi)
    cents_off = (midi - nearest_midi) * 100.0

    needs_corr = voiced & (np.abs(cents_off) > VOICE_DEADZONE_CENTS)
    excess_cents = np.sign(cents_off) * np.maximum(0.0, np.abs(cents_off) - VOICE_DEADZONE_CENTS)
    target_midi = midi - np.where(needs_corr, (excess_cents / 100.0) * VOICE_PITCH_STRENGTH, 0.0)

    alpha = 1.0 - np.exp(-5.0 / max(VOICE_ATTACK_MS, 1.0))
    smoothed_midi = midi.copy()
    for i in range(1, len(midi)):
        if voiced[i] and voiced[i - 1]:
            delta = target_midi[i] - midi[i]
            prev_delta = smoothed_midi[i - 1] - midi[i - 1]
            cur_delta = prev_delta + alpha * (delta - prev_delta)
            smoothed_midi[i] = midi[i] + cur_delta
        elif voiced[i]:
            smoothed_midi[i] = midi[i] + alpha * (target_midi[i] - midi[i])

    f0_corrected = f0.copy()
    f0_corrected[voiced] = 440.0 * (2.0 ** ((smoothed_midi[voiced] - 69.0) / 12.0))
    synthesized = pw.synthesize(f0_corrected, sp, ap, sr, frame_period=5.0)

    if len(synthesized) > len(segment):
        synthesized = synthesized[: len(segment)]
    elif len(synthesized) < len(segment):
        synthesized = np.pad(synthesized, (0, len(segment) - len(synthesized)))
    return synthesized.astype(np.float32)


def _apply_natural_pitch_voice(y: np.ndarray, sr: int) -> np.ndarray:
    chunk_samples = int(CHUNK_SECONDS * sr)
    overlap_samples = int((CROSSFADE_MS / 1000.0) * sr)

    if len(y) <= chunk_samples:
        return _pitch_correct_segment(y, sr)

    step = chunk_samples - overlap_samples
    output = np.zeros_like(y, dtype=np.float32)
    pos = 0
    first_block = True

    while pos < len(y):
        end = min(pos + chunk_samples, len(y))
        seg = y[pos:end]
        proc = _pitch_correct_segment(seg, sr)

        if first_block:
            output[pos:end] = proc
            first_block = False
        else:
            actual_overlap = min(overlap_samples, len(proc))
            if actual_overlap > 1:
                t = np.linspace(0.0, np.pi, actual_overlap, dtype=np.float32)
                fade_in = 0.5 * (1.0 - np.cos(t))
                fade_out = 1.0 - fade_in
                output[pos : pos + actual_overlap] = (
                    output[pos : pos + actual_overlap] * fade_out + proc[:actual_overlap] * fade_in
                )
                output[pos + actual_overlap : end] = proc[actual_overlap:]
            else:
                output[pos:end] = proc

        if end >= len(y):
            break
        pos += step

    return output


def _process_voice(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_cellphone_noise(y, sr, prop_decrease=0.55)
    y = _apply_natural_pitch_voice(y, sr)
    y = _deesser_soft(y, sr)
    board = Pedalboard([
        NoiseGate(threshold_db=-46.0, ratio=2.0, attack_ms=10.0, release_ms=120.0),
        HighpassFilter(cutoff_frequency_hz=85.0),
        LowShelfFilter(cutoff_frequency_hz=160.0, gain_db=2.2, q=0.7),
        PeakFilter(cutoff_frequency_hz=380.0, gain_db=-2.8, q=1.4),
        PeakFilter(cutoff_frequency_hz=2800.0, gain_db=-2.0, q=1.5),
        HighShelfFilter(cutoff_frequency_hz=9500.0, gain_db=2.0, q=0.7),
        Compressor(threshold_db=-20.0, ratio=2.0, attack_ms=18.0, release_ms=140.0),
        Reverb(room_size=0.18, damping=0.65, wet_level=0.07, dry_level=0.93, width=0.6),
        Limiter(threshold_db=-1.5, release_ms=80.0),
    ])
    return board(y, sr).astype(np.float32)


def _process_nylon(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_cellphone_noise(y, sr, prop_decrease=0.50)
    board = Pedalboard([
        NoiseGate(threshold_db=-48.0, ratio=1.8, attack_ms=12.0, release_ms=150.0),
        HighpassFilter(cutoff_frequency_hz=75.0),
        LowShelfFilter(cutoff_frequency_hz=180.0, gain_db=2.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=320.0, gain_db=-2.5, q=1.5),
        HighShelfFilter(cutoff_frequency_hz=6500.0, gain_db=-1.5, q=0.7),
        Compressor(threshold_db=-22.0, ratio=2.2, attack_ms=22.0, release_ms=160.0),
        Reverb(room_size=0.22, damping=0.60, wet_level=0.09, dry_level=0.91, width=0.7),
        Limiter(threshold_db=-1.5, release_ms=90.0),
    ])
    return board(y, sr).astype(np.float32)


def _process_aco(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_cellphone_noise(y, sr, prop_decrease=0.50)
    board = Pedalboard([
        NoiseGate(threshold_db=-48.0, ratio=1.8, attack_ms=10.0, release_ms=140.0),
        HighpassFilter(cutoff_frequency_hz=85.0),
        PeakFilter(cutoff_frequency_hz=260.0, gain_db=-2.5, q=1.5),
        PeakFilter(cutoff_frequency_hz=2400.0, gain_db=-1.5, q=1.6),
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=2.5, q=0.7),
        Compressor(threshold_db=-21.0, ratio=2.3, attack_ms=15.0, release_ms=140.0),
        Reverb(room_size=0.20, damping=0.55, wet_level=0.08, dry_level=0.92, width=0.75),
        Limiter(threshold_db=-1.5, release_ms=85.0),
    ])
    return board(y, sr).astype(np.float32)


def enhance_single_track(input_path: str, original_filename: str, matrix: str) -> tuple[str, str]:
    validate_not_already_treated(original_filename)

    matrix_clean = (matrix or "").strip().lower()
    if matrix_clean not in VALID_MATRICES:
        raise ValueError(f"Matriz inválida '{matrix}'. Use: {', '.join(VALID_MATRICES)}.")

    y, sr = load_audio(input_path, target_sr=SAMPLE_RATE, mono=True)

    if matrix_clean == "voz":
        processed = _process_voice(y, sr)
    elif matrix_clean == "nylon":
        processed = _process_nylon(y, sr)
    else:
        processed = _process_aco(y, sr)

    stereo_out = np.column_stack([processed, processed])
    out_filename = build_clean_treated_filename(original_filename, matrix_clean)
    out_path = os.path.join(ensure_temp_dir(), out_filename)
    save_wav(out_path, stereo_out, sr)
    return out_path, out_filename
