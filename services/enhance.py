import os
import uuid
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
    Limiter,
)
from config import (
    SAMPLE_RATE,
    VALID_MATRICES,
    VOICE_PITCH_STRENGTH,
    VOICE_DEADZONE_CENTS,
    VOICE_ATTACK_MS,
)
from services.audio_utils import load_audio, save_wav, normalize_lufs, extract_track_name, ensure_temp_dir
from services.references import get_reference_profile, analyze_tonal_profile


def validate_not_already_treated(filename: str) -> None:
    base = os.path.basename(filename or "").upper()
    if base.startswith("PRE-MIX+_") or base.startswith("PRE-MIX_") or base.startswith("MIX_"):
        raise ValueError("Faixa já foi tratada (prefixo PRE-MIX+_ detectado). Não é permitido retratar.")


def _reduce_noise_clean(
    y: np.ndarray,
    sr: int,
    prop_decrease: float = 0.72,
    n_std_thresh_stationary: float = 1.25,
) -> np.ndarray:
    """Remoção de ruído mais firme mantendo naturalidade do sinal."""
    if len(y) < int(sr * 0.5):
        return y.astype(np.float32)

    cleaned = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=prop_decrease,
        n_std_thresh_stationary=n_std_thresh_stationary,
    ).astype(np.float32)

    # Passada complementar suave não-estacionária para ruídos/estalos variáveis
    non_stat = nr.reduce_noise(
        y=cleaned,
        sr=sr,
        stationary=False,
        prop_decrease=min(0.45, prop_decrease * 0.6),
        n_std_thresh_nonstationary=1.5,
    ).astype(np.float32)

    # Blend 70% não-estacionário refinado + 30% estacionário limpo para evitar artefatos metálicos
    blended = (0.70 * non_stat + 0.30 * cleaned).astype(np.float32)
    return blended


def _deesser_soft(y: np.ndarray, sr: int, threshold: float = 0.065, ratio: float = 0.60) -> np.ndarray:
    sos = butter(4, [5200, 9800], btype="bandpass", fs=sr, output="sos")
    sibilance = sosfilt(sos, y)
    env = np.abs(sibilance)
    b, a = butter(1, 45.0 / (sr / 2.0), btype="low")
    smooth_env = lfilter(b, a, env)
    gain = np.where(
        smooth_env > threshold,
        1.0 - ratio * np.clip((smooth_env - threshold) / (threshold + 1e-6), 0.0, 1.0),
        1.0,
    )
    return (y - sibilance * (1.0 - gain)).astype(np.float32)


def _pitch_correct_voice(y: np.ndarray, sr: int) -> np.ndarray:
    y64 = y.astype(np.float64)
    f0, t = pw.harvest(y64, sr, f0_floor=75.0, f0_ceil=650.0, frame_period=5.0)
    f0 = pw.stonemask(y64, f0, t, sr)
    sp = pw.cheaptrick(y64, f0, t, sr)
    ap = pw.d4c(y64, f0, t, sr)

    voiced = f0 > 0
    if not np.any(voiced):
        return y.astype(np.float32)

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
            smoothed_midi[i] = midi[i] + prev_delta + alpha * (delta - prev_delta)
        elif voiced[i]:
            smoothed_midi[i] = midi[i] + alpha * (target_midi[i] - midi[i])

    f0_corrected = f0.copy()
    f0_corrected[voiced] = 440.0 * (2.0 ** ((smoothed_midi[voiced] - 69.0) / 12.0))
    synthesized = pw.synthesize(f0_corrected, sp, ap, sr, frame_period=5.0)
    if len(synthesized) > len(y):
        synthesized = synthesized[: len(y)]
    elif len(synthesized) < len(y):
        synthesized = np.pad(synthesized, (0, len(y) - len(synthesized)))
    return synthesized.astype(np.float32)


def _apply_reference_match(y: np.ndarray, sr: int, ref_profile: dict) -> np.ndarray:
    cur = analyze_tonal_profile(y, sr)

    def _clamped_delta(key: str, max_db: float = 4.0) -> float:
        diff = float(ref_profile.get(key, cur[key])) - float(cur[key])
        return float(np.clip(diff * 0.55, -max_db, max_db))

    match_board = Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=_clamped_delta("low_rel_db", 4.0), q=0.7),
        PeakFilter(cutoff_frequency_hz=2600.0, gain_db=_clamped_delta("pres_rel_db", 3.5), q=1.1),
        HighShelfFilter(cutoff_frequency_hz=8500.0, gain_db=_clamped_delta("air_rel_db", 4.0), q=0.7),
        Limiter(threshold_db=-1.2, release_ms=80.0),
    ])
    return match_board(y, sr).astype(np.float32)


def _apply_puigtec_analog_eq(y: np.ndarray, sr: int, matrix: str) -> np.ndarray:
    """
    Emulação analógica inspirada nos plugins Waves PuigTec EQP-1A & MEQ-5:
    - Pultec Low-End Trick (Boost + Atten simultâneos nos graves para peso sem lama)
    - Curvas musicais largas do MEQ-5 nos médios
    - High-Shelf sedoso com atenuação complementar (Atten Sel) nos agudos
    - Saturação harmônica assimétrica de estágio valvulado (Tube Output Stage)
    """
    m = matrix.lower()
    if m == "voz":
        eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=85.0),
            LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=3.2, q=0.65),
            PeakFilter(cutoff_frequency_hz=215.0, gain_db=-3.2, q=1.1),
            PeakFilter(cutoff_frequency_hz=380.0, gain_db=-3.5, q=1.25),
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=3.2, q=0.9),
            HighShelfFilter(cutoff_frequency_hz=12000.0, gain_db=4.2, q=0.65),
            PeakFilter(cutoff_frequency_hz=18000.0, gain_db=-1.5, q=1.2),
        ])
        drive = 1.18
    elif m == "nylon":
        eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=75.0),
            LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=3.4, q=0.65),
            PeakFilter(cutoff_frequency_hz=240.0, gain_db=-3.8, q=1.15),
            PeakFilter(cutoff_frequency_hz=520.0, gain_db=-2.4, q=1.0),
            PeakFilter(cutoff_frequency_hz=2500.0, gain_db=2.6, q=0.85),
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=3.4, q=0.65),
            PeakFilter(cutoff_frequency_hz=16000.0, gain_db=-1.2, q=1.2),
        ])
        drive = 1.15
    elif m == "aco":
        eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80.0),
            LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=3.0, q=0.65),
            PeakFilter(cutoff_frequency_hz=250.0, gain_db=-3.8, q=1.15),
            PeakFilter(cutoff_frequency_hz=420.0, gain_db=-2.6, q=1.1),
            PeakFilter(cutoff_frequency_hz=4000.0, gain_db=3.0, q=0.9),
            PeakFilter(cutoff_frequency_hz=6800.0, gain_db=-2.2, q=1.7),
            HighShelfFilter(cutoff_frequency_hz=12000.0, gain_db=4.0, q=0.65),
        ])
        drive = 1.16
    elif m == "baixo":
        eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=35.0),
            LowShelfFilter(cutoff_frequency_hz=60.0, gain_db=4.6, q=0.65),
            PeakFilter(cutoff_frequency_hz=135.0, gain_db=-3.8, q=1.15),
            PeakFilter(cutoff_frequency_hz=300.0, gain_db=-2.8, q=1.2),
            PeakFilter(cutoff_frequency_hz=850.0, gain_db=3.2, q=0.95),
            PeakFilter(cutoff_frequency_hz=2800.0, gain_db=2.2, q=1.0),
            HighShelfFilter(cutoff_frequency_hz=5500.0, gain_db=-2.8, q=0.7),
        ])
        drive = 1.20
    else:  # solo
        eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=85.0),
            LowShelfFilter(cutoff_frequency_hz=110.0, gain_db=2.6, q=0.7),
            PeakFilter(cutoff_frequency_hz=270.0, gain_db=-3.6, q=1.15),
            PeakFilter(cutoff_frequency_hz=1500.0, gain_db=2.6, q=0.9),
            PeakFilter(cutoff_frequency_hz=3200.0, gain_db=3.6, q=0.9),
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=3.5, q=0.65),
            PeakFilter(cutoff_frequency_hz=16000.0, gain_db=-1.5, q=1.2),
        ])
        drive = 1.17

    shaped = eq(y, sr).astype(np.float32)

    # Estágio de saída valvulado PuigTec (harmônicos pares + ímpares suaves sem clipar)
    tube_bias = 0.024
    warm = (np.tanh((shaped + tube_bias) * drive) - np.tanh(tube_bias * drive)) / np.tanh(drive)
    return warm.astype(np.float32)


def _process_voice(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_noise_clean(y, sr, prop_decrease=0.75, n_std_thresh_stationary=1.20)
    y = _pitch_correct_voice(y, sr)
    y = _deesser_soft(y, sr, threshold=0.065, ratio=0.60)
    y = Pedalboard([NoiseGate(threshold_db=-42.0, ratio=3.0, attack_ms=6.0, release_ms=110.0)])(y, sr)
    y = _apply_puigtec_analog_eq(y, sr, "voz")
    y = Pedalboard([
        Compressor(threshold_db=-18.0, ratio=3.2, attack_ms=12.0, release_ms=120.0),
        Limiter(threshold_db=-1.2, release_ms=80.0),
    ])(y, sr)
    return y.astype(np.float32)


def _process_nylon(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_noise_clean(y, sr, prop_decrease=0.68, n_std_thresh_stationary=1.25)
    y = Pedalboard([NoiseGate(threshold_db=-45.0, ratio=2.6, attack_ms=8.0, release_ms=140.0)])(y, sr)
    y = _apply_puigtec_analog_eq(y, sr, "nylon")
    y = Pedalboard([
        Compressor(threshold_db=-19.5, ratio=2.8, attack_ms=16.0, release_ms=140.0),
        Limiter(threshold_db=-1.2, release_ms=85.0),
    ])(y, sr)
    return y.astype(np.float32)


def _process_aco(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_noise_clean(y, sr, prop_decrease=0.70, n_std_thresh_stationary=1.25)
    y = Pedalboard([NoiseGate(threshold_db=-44.0, ratio=2.8, attack_ms=8.0, release_ms=130.0)])(y, sr)
    y = _apply_puigtec_analog_eq(y, sr, "aco")
    y = Pedalboard([
        Compressor(threshold_db=-19.0, ratio=3.0, attack_ms=12.0, release_ms=130.0),
        Limiter(threshold_db=-1.2, release_ms=80.0),
    ])(y, sr)
    return y.astype(np.float32)


def _process_baixo(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_noise_clean(y, sr, prop_decrease=0.60, n_std_thresh_stationary=1.30)
    y = Pedalboard([NoiseGate(threshold_db=-46.0, ratio=2.6, attack_ms=8.0, release_ms=120.0)])(y, sr)
    y = _apply_puigtec_analog_eq(y, sr, "baixo")
    y = Pedalboard([
        Compressor(threshold_db=-17.5, ratio=3.8, attack_ms=12.0, release_ms=110.0),
        Limiter(threshold_db=-1.2, release_ms=80.0),
    ])(y, sr)
    return y.astype(np.float32)


def _process_solo(y: np.ndarray, sr: int) -> np.ndarray:
    y = _reduce_noise_clean(y, sr, prop_decrease=0.68, n_std_thresh_stationary=1.25)
    y = Pedalboard([NoiseGate(threshold_db=-44.0, ratio=2.6, attack_ms=8.0, release_ms=130.0)])(y, sr)
    y = _apply_puigtec_analog_eq(y, sr, "solo")
    y = Pedalboard([
        Compressor(threshold_db=-19.0, ratio=3.0, attack_ms=12.0, release_ms=120.0),
        Limiter(threshold_db=-1.2, release_ms=80.0),
    ])(y, sr)
    return y.astype(np.float32)


def enhance_single_track(
    input_path: str,
    original_filename: str,
    matrix: str,
    use_reference: bool = True,
) -> tuple[str, str, bool]:
    validate_not_already_treated(original_filename)

    matrix_clean = (matrix or "").strip().lower()
    if matrix_clean == "aço":
        matrix_clean = "aco"
    if matrix_clean not in VALID_MATRICES:
        raise ValueError(f"Matriz inválida '{matrix}'. Use: {', '.join(VALID_MATRICES)}.")

    y, sr = load_audio(input_path, target_sr=SAMPLE_RATE, mono=True)

    if matrix_clean == "voz":
        processed = _process_voice(y, sr)
    elif matrix_clean == "nylon":
        processed = _process_nylon(y, sr)
    elif matrix_clean == "aco":
        processed = _process_aco(y, sr)
    elif matrix_clean == "baixo":
        processed = _process_baixo(y, sr)
    else:
        processed = _process_solo(y, sr)

    reference_applied = False
    if use_reference:
        ref_profile = get_reference_profile(matrix_clean)
        if ref_profile is not None:
            processed = _apply_reference_match(processed, sr, ref_profile)
            reference_applied = True

    stereo_out = normalize_lufs(np.column_stack([processed, processed]), sr=sr)
    out_filename = f"PRE-MIX+_{extract_track_name(original_filename)}.wav"
    out_path = os.path.join(ensure_temp_dir(), f"tmp_{uuid.uuid4().hex[:6]}_{out_filename}")
    save_wav(out_path, stereo_out, sr)
    return out_path, out_filename, reference_applied
