import os
import math
import uuid
import numpy as np
from pedalboard import Pedalboard, Reverb
from config import SAMPLE_RATE, MIX_PAN_AND_GAIN, PLAYBACK_GAIN_DB, VALID_MATRICES
from services.audio_utils import load_audio, save_wav, normalize_lufs, extract_track_name, ensure_temp_dir


def validate_treated_filenames(filenames: list[str]) -> None:
    if len(filenames) < 2:
        raise ValueError("A mixagem exige pelo menos 2 faixas já tratadas (prefixo PRE-MIX+_).")
    for name in filenames:
        upper_name = os.path.basename(name).upper()
        is_treated = upper_name.startswith("PRE-MIX+_") or upper_name.startswith("PRE-MIX_")
        if upper_name.startswith("GRAVAÇÃO_") or upper_name.startswith("GRAVACAO_") or not is_treated:
            raise ValueError(
                f"Arquivo inválido na mixagem: '{name}'. Apenas faixas tratadas (PRE-MIX+_) são aceitas."
            )


def _detect_type_from_filename(filename: str) -> str | None:
    lower = os.path.basename(filename).lower()
    tokens = lower.replace("-", "_").replace(".", "_").replace("+", "_").split("_")
    for t in tokens:
        if t in ("voz", "vocal", "vocals"):
            return "voz"
        if t == "nylon":
            return "nylon"
        if t in ("aco", "aço", "violao", "violão"):
            return "aco"
        if t in ("baixo", "bass"):
            return "baixo"
        if t in ("solo", "lead"):
            return "solo"
    return None


def resolve_track_types(filenames: list[str], provided_types: list[str] | None = None) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    sources: list[str] = []
    valid = set(VALID_MATRICES)

    for idx, fname in enumerate(filenames):
        raw = ""
        if provided_types and idx < len(provided_types):
            raw = (provided_types[idx] or "").strip().lower()
            if raw == "aço":
                raw = "aco"

        if raw in valid:
            resolved.append(raw)
            sources.append(f"{os.path.basename(fname)} -> {raw}")
            continue

        detected = _detect_type_from_filename(fname)
        if detected:
            resolved.append(detected)
            sources.append(f"{os.path.basename(fname)} -> {detected}")
        else:
            fallback = "voz" if idx == 0 else "aco"
            resolved.append(fallback)
            sources.append(f"{os.path.basename(fname)} -> {fallback}")
    return resolved, sources


def _apply_pan_gain_and_reverb(audio: np.ndarray, t_type: str, pan: float, gain_db: float) -> np.ndarray:
    if t_type == "voz":
        audio = Pedalboard([Reverb(room_size=0.22, damping=0.60, wet_level=0.08, dry_level=0.92, width=0.65)])(
            audio.T, SAMPLE_RATE
        ).T
    elif t_type in ("nylon", "aco", "solo"):
        audio = Pedalboard([Reverb(room_size=0.26, damping=0.58, wet_level=0.08, dry_level=0.92, width=0.75)])(
            audio.T, SAMPLE_RATE
        ).T

    pan_clamped = max(-1.0, min(1.0, float(pan)))
    gain_lin = 10.0 ** (float(gain_db) / 20.0)
    angle = (pan_clamped + 1.0) * (math.pi / 4.0)
    left_gain = min(1.0, math.cos(angle) * math.sqrt(2.0)) * gain_lin
    right_gain = min(1.0, math.sin(angle) * math.sqrt(2.0)) * gain_lin

    out = np.empty_like(audio, dtype=np.float32)
    out[:, 0] = audio[:, 0] * left_gain
    out[:, 1] = audio[:, 1] * right_gain
    return out


def mix_treated_tracks(
    track_paths: list[str],
    original_filenames: list[str],
    track_types: list[str] | None = None,
    guitar_roles: list[str] | None = None,
    playback_path: str | None = None,
    use_playback: bool = False,
    project_name: str | None = None,
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
        role = (guitar_roles[idx] or "").strip().upper() if (guitar_roles and idx < len(guitar_roles)) else ""

        if t_type in ("voz", "baixo", "solo"):
            cfg = MIX_PAN_AND_GAIN[t_type]
        else:
            role_key = role if role in ("BASE", "SOLO") else "DEFAULT"
            cfg = MIX_PAN_AND_GAIN[t_type][role_key]

        positioned = _apply_pan_gain_and_reverb(audio, t_type, cfg["pan"], cfg["gain_db"])
        mix_bus[: positioned.shape[0]] += positioned

    if use_playback and playback_path:
        pb_audio, _ = load_audio(playback_path, target_sr=SAMPLE_RATE, mono=False)
        pb_gain = 10.0 ** (PLAYBACK_GAIN_DB / 20.0)
        pb_trimmed = pb_audio[:max_treated_len] * pb_gain
        mix_bus[: pb_trimmed.shape[0]] += pb_trimmed

    mastered = normalize_lufs(mix_bus, sr=SAMPLE_RATE)
    base_name = extract_track_name(project_name.strip()) if (project_name and project_name.strip()) else extract_track_name(original_filenames[0])
    out_filename = f"MIX_{base_name}.wav"
    out_path = os.path.join(ensure_temp_dir(), f"tmp_{uuid.uuid4().hex[:6]}_{out_filename}")
    save_wav(out_path, mastered, SAMPLE_RATE)
    return out_path, out_filename, resolution_notes
