import os

STUDIO_NAME = "render-base"
TEMP_DIR = "/tmp/render_studio"

SAMPLE_RATE = 44100
TARGET_LUFS = -14.0
TRUE_PEAK_CEILING_DB = -1.0

VALID_MATRICES = ("voz", "nylon", "aco")

VOICE_PITCH_STRENGTH = 0.35
VOICE_DEADZONE_CENTS = 25.0
VOICE_ATTACK_MS = 180.0

MIX_PAN_AND_GAIN = {
    "voz": {"pan": 0.0, "gain_db": 0.0},
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
