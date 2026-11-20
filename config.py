import os

STUDIO_NAME = os.getenv("STUDIO_NAME", "render-base")
STUDIO_DISPLAY_NAME = os.getenv("STUDIO_DISPLAY_NAME", "Render Studio - PuigTec Free")

TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/render_studio")
REFERENCES_DIR = os.getenv("REFERENCES_DIR", os.path.join(os.getcwd(), "data", "references"))

SAMPLE_RATE = 44100
TARGET_LUFS = -14.0
TRUE_PEAK_CEILING_DB = -1.0

VALID_MATRICES = ("voz", "nylon", "aco", "baixo", "solo")

MATRIX_LABELS = {
    "voz": "Voz",
    "nylon": "Violão Nylon",
    "aco": "Violão Aço",
    "baixo": "Baixo",
    "solo": "Solo",
}

VOICE_PITCH_STRENGTH = 0.35
VOICE_DEADZONE_CENTS = 25.0
VOICE_ATTACK_MS = 180.0

MIX_PAN_AND_GAIN = {
    "voz": {"pan": 0.0, "gain_db": 0.0},
    "baixo": {"pan": 0.0, "gain_db": -1.0},
    "solo": {"pan": 0.0, "gain_db": -0.5},
    "nylon": {
        "BASE": {"pan": -0.35, "gain_db": -2.5},
        "SOLO": {"pan": 0.0, "gain_db": -0.5},
        "DEFAULT": {"pan": -0.25, "gain_db": -2.0},
    },
    "aco": {
        "BASE": {"pan": 0.35, "gain_db": -2.5},
        "SOLO": {"pan": 0.0, "gain_db": -0.5},
        "DEFAULT": {"pan": 0.25, "gain_db": -2.0},
    },
}

PLAYBACK_GAIN_DB = 0.0
