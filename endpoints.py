API_VERSION = "3.1.0-puigtec"

NAMING_RULES = {
    "raw_prefix": "Gravação_",
    "treated_prefix": "PRE-MIX+_",
    "final_mix_prefix": "MIX_",
    "reference_prefix": "ref_",
}

ENDPOINTS = [
    {"method": "GET", "path": "/health", "description": "Verifica status online do Studio"},
    {"method": "GET", "path": "/info", "description": "Descoberta de matrizes e status das 5 referências"},
    {"method": "GET", "path": "/endpoints", "description": "Lista contrato universal de endpoints"},
    {"method": "GET", "path": "/references/status", "description": "Status das 5 referências (present/absent)"},
    {"method": "POST", "path": "/references/{matrix}", "description": "Envia arquivo de referência (voz, nylon, aco, baixo, solo)"},
    {"method": "DELETE", "path": "/references/{matrix}", "description": "Remove arquivo de referência"},
    {"method": "POST", "path": "/enhance", "description": "Trata 1 faixa Gravação_ e devolve PRE-MIX+_ (Matriz Seca PuigTec)"},
    {"method": "POST", "path": "/mixar-final", "description": "Mixa 2+ faixas PRE-MIX+_ (+ playback opcional) -> MIX_"},
]
