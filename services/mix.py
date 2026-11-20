import os
import math
from datetime import datetime
import numpy as np
from config import SAMPLE_RATE, MIX_PAN_AND_GAIN, PLAYBACK_GAIN_DB
from services.audio_utils import load_audio, save_wav, normalize_lufs, ensure_temp_dir


def validate_treated_filenames(filenames: list[str]) -> None:
    if len(filenames) < 2:
        raise ValueError("A mixagem exige pelo menos 2 faixas já tratadas (prefixo iATRAT_).")
    for name in filenames:
        upper_name = os.path.basename(name).upper()
        if upper_name.startswith("IAREC_") or not upper_name.startswith("IATRAT_"):
            raise ValueError(
                f"Arquivo inválido na mixagem: '{name}'. Apenas faixas tratadas (iATRAT_) são aceitas."
            )


def _detect_type_from_filename(filename: str) -> str | None:
    lower = os.path.basename(filename).lower()
    tokens = lower.replace("-", "_").replace(".", "_").split("_")
    for t in tokens:
        if t == "voz":
            return "voz"
        if t == "nylon":
            return "nylon"
        if t in ("aco", "aço"):
            return "aco"
    return None


def resolve_track_types(
    filenames: list[str],
    provided_types: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    sources: list[str] = []
    valid = {"voz", "nylon", "aco"}

    for idx, fname in enumerate(filenames):
        raw = ""
        if provided_types and idx < len(provided_types):
            raw = (provided_types[idx] or "").strip().lower()
            if raw == "aço":
                raw = "aco"

        if raw in valid:
            resolved.append(raw)
            sources.append(f"{os.path.basename(fname)} -> {raw} (enviado pelo app)")
            continue

        detected = _detect_type_from_filename(fname)
        if detected:
            resolved.append(detected)
            sources.append(f"{os.path.basename(fname)} -> {detected} (detectado pelo nome)")
        else:
            fallback = "voz" if idx == 0 else "aco"
            resolved.append(fallback)
            reason = "fallback 1a faixa" if idx == 0 else "fallback instrumento"
            sources.append(f"{os.path.basename(fname)} -> {fallback} ({reason})")

    return resolved, sources


def _apply_pan_and_gain(audio: np.ndarray, pan: float, gain_db: float) -> np.ndarray:
    pan_clamped = max(-1.0, min(1.0, float(pan)))
    gain_lin = 10.0 ** (float(gain_db) / 20.0)

    angle = (pan_clamped + 1.0) * (math.pi / 4.0)
    left_gain = min(1.0, math.cos(angle) * math.sqrt(2.0)) * gain_lin
    right_gain = min(1.0, math.sin(angle) * math.sqrt(2.0)) * gain_lin

    out = np.empty_like(audio, dtype=np.float32)
    out[:, 0] = audio[:, 0] * left_gain
    out[:, 1] = audio[:, 1] * right_gain
    return out


def get_next_mix_filename(output_dir: str) -> tuple[str, str]:
    date_str = datetime.now().strftime("%Y-%m-%d")
    seq = 1
    while True:
        filename = f"iAMIX_{date_str}_{seq:02d}.wav"
        full_path = os.path.join(output_dir, filename)
        if not os.path.exists(full_path):
            return full_path, filename
        seq += 1


def mix_treated_tracks(
    track_paths: list[str],
    original_filenames: list[str],
    track_types: list[str] | None = None,
    guitar_roles: list[str] | None = None,
    playback_path: str | None = None,
    use_playback: bool = False,
) -> tuple[str, str, list[str]]:
    validate_treated_filenames(original_filenames)
    resolved_types, resolution_notes = resolve_track_types(original_filenames, track_types)

    loaded_tracks: list[np.ndarray] = []
    for p in track_paths:
        audio, _ = load_audio(p, target_sr=SAMPLE_RATE, mono=False)
        loaded_tracks.append(audio)

    max_treated_len = max(t.shape[0] for t in loaded_tracks)
    mix_bus = np.zeros((max_treated_len, 2), dtype=np.float32)

    for idx, audio in enumerate(loaded_tracks):
        t_type = resolved_types[idx]
        role = ""
        if guitar_roles and idx < len(guitar_roles):
            role = (guitar_roles[idx] or "").strip().upper()

        if t_type == "voz":
            cfg = MIX_PAN_AND_GAIN["voz"]
        else:
            role_key = role if role in ("BASE", "SOLO") else "DEFAULT"
            cfg = MIX_PAN_AND_GAIN[t_type][role_key]

        positioned = _apply_pan_and_gain(audio, cfg["pan"], cfg["gain_db"])
        mix_bus[: positioned.shape[0]] += positioned

    if use_playback and playback_path:
        pb_audio, _ = load_audio(playback_path, target_sr=SAMPLE_RATE, mono=False)
        pb_gain = 10.0 ** (PLAYBACK_GAIN_DB / 20.0)
        pb_trimmed = pb_audio[:max_treated_len] * pb_gain
        mix_bus[: pb_trimmed.shape[0]] += pb_trimmed

    mastered = normalize_lufs(mix_bus, sr=SAMPLE_RATE)
    out_dir = ensure_temp_dir()
    out_path, out_filename = get_next_mix_filename(out_dir)
    save_wav(out_path, mastered, SAMPLE_RATE)
    return out_path, out_filename, resolution_notes
