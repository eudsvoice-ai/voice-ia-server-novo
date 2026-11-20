# Render Studio - IA Base (v2)

## Rotas
- `GET /` -> Status textual
- `GET /health` -> `{"status": "ok", "studio": "render-base"}`
- `POST /enhance` -> Trata 1 faixa (`voz`, `nylon`, `aco`) e devolve `iATRAT_<matrix>_<nome>.wav`
- `POST /mixar-final` -> Mixa 2+ faixas `iATRAT_` + playback opcional (0 dB, cortado no tamanho da maior faixa tratada) em `-14 LUFS` estéreo e devolve `iAMIX_YYYY-MM-DD_01.wav`.
