import os
import json
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, FileResponse
from config import STUDIO_NAME
from services.audio_utils import ensure_temp_dir, cleanup_files
from services.enhance import enhance_single_track
from services.mix import mix_treated_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("render-studio")

app = FastAPI(title="Render Studio - IA Base", version="2.0.0")


def _parse_list_field(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(",")]


@app.get("/", response_class=PlainTextResponse)
def root():
    return "Render Studio IA Base - OK"


@app.get("/health")
def health():
    return {"status": "ok", "studio": STUDIO_NAME}


@app.post("/enhance")
async def enhance_route(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    matrix: str = Form(...),
):
    if len(files) != 1:
        raise HTTPException(
            status_code=400,
            detail="A rota /enhance aceita exatamente 1 arquivo por vez.",
        )

    upload = files[0]
    original_name = upload.filename or "audio.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (".wav", ".mp3"):
        raise HTTPException(status_code=400, detail="Formato inválido. Envie WAV ou MP3.")

    temp_dir = ensure_temp_dir()
    in_path = os.path.join(temp_dir, f"in_{uuid.uuid4().hex[:8]}{ext}")
    temp_files = [in_path]

    try:
        content = await upload.read()
        with open(in_path, "wb") as f:
            f.write(content)

        out_path, out_filename = enhance_single_track(in_path, original_name, matrix)
        temp_files.append(out_path)
        logger.info(f"Enhance {matrix.lower()} OK -> {out_filename}")

        background_tasks.add_task(cleanup_files, temp_files)
        return FileResponse(
            path=out_path,
            media_type="audio/wav",
            filename=out_filename,
            headers={
                "X-Matrix-Used": matrix.lower(),
                "X-Output-Filename": out_filename,
            },
        )
    except ValueError as e:
        cleanup_files(temp_files)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        cleanup_files(temp_files)
        logger.error(f"Erro no /enhance: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar áudio: {str(e)}")


@app.post("/mixar-final")
async def mixar_final_route(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    playback: UploadFile | None = File(None),
    use_playback: bool = Form(False),
    track_types: str | None = Form(None),
    guitar_roles: str | None = Form(None),
):
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Envie pelo menos 2 faixas tratadas (iATRAT_) para mixar.",
        )

    temp_dir = ensure_temp_dir()
    input_temp_files: list[str] = []
    track_paths: list[str] = []
    original_names: list[str] = []

    try:
        for idx, up in enumerate(files):
            orig = up.filename or f"track_{idx}.wav"
            ext = os.path.splitext(orig)[1].lower() or ".wav"
            tmp_in = os.path.join(temp_dir, f"mix_in_{idx}_{uuid.uuid4().hex[:8]}{ext}")
            data = await up.read()
            with open(tmp_in, "wb") as f:
                f.write(data)
            input_temp_files.append(tmp_in)
            track_paths.append(tmp_in)
            original_names.append(orig)

        pb_path = None
        if use_playback and playback is not None and playback.filename:
            pb_ext = os.path.splitext(playback.filename)[1].lower() or ".wav"
            pb_path = os.path.join(temp_dir, f"pb_{uuid.uuid4().hex[:8]}{pb_ext}")
            pb_data = await playback.read()
            with open(pb_path, "wb") as f:
                f.write(pb_data)
            input_temp_files.append(pb_path)

        parsed_types = _parse_list_field(track_types)
        parsed_roles = _parse_list_field(guitar_roles)

        out_path, out_filename, resolution_notes = mix_treated_tracks(
            track_paths=track_paths,
            original_filenames=original_names,
            track_types=parsed_types,
            guitar_roles=parsed_roles,
            playback_path=pb_path,
            use_playback=use_playback,
        )

        notes_summary = " | ".join(resolution_notes)
        logger.info(f"Mix {len(files)} faixas OK -> {out_filename} [{notes_summary}]")

        background_tasks.add_task(cleanup_files, input_temp_files)
        return FileResponse(
            path=out_path,
            media_type="audio/wav",
            filename=out_filename,
            headers={
                "X-Output-Filename": out_filename,
                "X-Resolved-Types": notes_summary,
            },
        )
    except ValueError as e:
        cleanup_files(input_temp_files)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        cleanup_files(input_temp_files)
        logger.error(f"Erro no /mixar-final: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao mixar faixas: {str(e)}")
