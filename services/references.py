import os
import json
from datetime import datetime, timezone
import numpy as np
from scipy.signal import butter, sosfilt
from config import VALID_MATRICES, MATRIX_LABELS, SAMPLE_RATE
from services.audio_utils import ensure_references_dir, load_audio, save_wav


def _normalize_matrix(matrix: str) -> str:
    m = (matrix or "").strip().lower()
    if m == "aço":
        m = "aco"
    if m not in VALID_MATRICES:
        raise ValueError(f"Matriz inválida '{matrix}'. Use: {', '.join(VALID_MATRICES)}.")
    return m


def _ref_paths(matrix: str) -> tuple[str, str]:
    m = _normalize_matrix(matrix)
    ref_dir = ensure_references_dir()
    return os.path.join(ref_dir, f"ref_{m}.wav"), os.path.join(ref_dir, f"ref_{m}.json")


def _band_energy_db(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    nyq = sr / 2.0
    hi = min(high_hz, nyq * 0.95)
    lo = max(20.0, low_hz)
    if lo >= hi:
        return -60.0
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    filtered = sosfilt(sos, y)
    rms = float(np.sqrt(np.mean(np.square(filtered)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def analyze_tonal_profile(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    rms_total = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    rms_db = 20.0 * np.log10(rms_total + 1e-12)
    return {
        "rms_db": round(rms_db, 2),
        "low_rel_db": round(_band_energy_db(y, sr, 60.0, 220.0) - rms_db, 2),
        "mid_rel_db": round(_band_energy_db(y, sr, 250.0, 1200.0) - rms_db, 2),
        "pres_rel_db": round(_band_energy_db(y, sr, 1500.0, 4500.0) - rms_db, 2),
        "air_rel_db": round(_band_energy_db(y, sr, 5000.0, 14000.0) - rms_db, 2),
    }


def save_reference_audio(input_path: str, matrix: str, original_filename: str = "") -> dict:
    m = _normalize_matrix(matrix)
    wav_path, meta_path = _ref_paths(m)
    y, sr = load_audio(input_path, target_sr=SAMPLE_RATE, mono=True)
    if y.size < sr * 0.5:
        raise ValueError("Áudio de referência muito curto (mínimo 0.5s).")

    if len(y) > sr * 60:
        y = y[: sr * 60]

    save_wav(wav_path, y, sr)
    profile = analyze_tonal_profile(y, sr)

    meta = {
        "matrix": m,
        "label": MATRIX_LABELS.get(m, m.upper()),
        "stored_filename": f"ref_{m}.wav",
        "original_filename": os.path.basename(original_filename) if original_filename else f"ref_{m}.wav",
        "duration_sec": round(len(y) / sr, 2),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def get_reference_profile(matrix: str) -> dict | None:
    m = _normalize_matrix(matrix)
    wav_path, meta_path = _ref_paths(m)
    if not os.path.exists(wav_path):
        return None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f).get("profile")
        except Exception:
            pass
    y, sr = load_audio(wav_path, target_sr=SAMPLE_RATE, mono=True)
    return analyze_tonal_profile(y, sr)


def delete_reference(matrix: str) -> bool:
    m = _normalize_matrix(matrix)
    wav_path, meta_path = _ref_paths(m)
    removed = False
    for p in (wav_path, meta_path):
        if os.path.exists(p):
            os.remove(p)
            removed = True
    return removed


def get_all_references_status() -> dict:
    items = {}
    for m in VALID_MATRICES:
        wav_path, meta_path = _ref_paths(m)
        if os.path.exists(wav_path):
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
            items[m] = {
                "matrix": m,
                "label": MATRIX_LABELS.get(m, m.upper()),
                "status": "present",
                "present": True,
                "message": f"Ativa ({meta.get('original_filename', f'ref_{m}.wav')})",
                "stored_filename": f"ref_{m}.wav",
                "original_filename": meta.get("original_filename", f"ref_{m}.wav"),
                "duration_sec": meta.get("duration_sec"),
                "updated_at": meta.get("updated_at"),
            }
        else:
            items[m] = {
                "matrix": m,
                "label": MATRIX_LABELS.get(m, m.upper()),
                "status": "absent",
                "present": False,
                "message": "Ausente - ainda não enviada (usando preset PuigTec Tube)",
                "stored_filename": f"ref_{m}.wav",
                "original_filename": None,
                "duration_sec": None,
                "updated_at": None,
            }
    return items
