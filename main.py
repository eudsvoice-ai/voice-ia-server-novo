import os
import json
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from config import STUDIO_NAME, STUDIO_DISPLAY_NAME, VALID_MATRICES, MATRIX_LABELS, SAMPLE_RATE, TARGET_LUFS
from endpoints import API_VERSION, ENDPOINTS, NAMING_RULES
from services.audio_utils import ensure_temp_dir, cleanup_files
from services.enhance import enhance_single_track
from services.mix import mix_treated_tracks
from services.references import save_reference_audio, get_all_references_status, delete_reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(STUDIO_NAME)

app = FastAPI(title=STUDIO_DISPLAY_NAME, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Matrix-Used",
        "X-Reference-Applied",
        "X-Output-Filename",
        "X-Resolved-Types",
        "Content-Disposition",
    ],
)


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
    return f"{STUDIO_DISPLAY_NAME} ({STUDIO_NAME}) - PuigTec Tube Edition - OK"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "studio": STUDIO_NAME,
        "display_name": STUDIO_DISPLAY_NAME,
        "version": API_VERSION,
    }


@app.get("/info")
def studio_info():
    return {
        "studio": STUDIO_NAME,
        "display_name": STUDIO_DISPLAY_NAME,
        "version": API_VERSION,
        "sample_rate": SAMPLE_RATE,
        "target_lufs": TARGET_LUFS,
        "matrices": [{"id": m, "label": MATRIX_LABELS.get(m, m.upper())} for m in VALID_MATRICES],
        "naming_rules": NAMING_RULES,
        "references": get_all_references_status(),
    }


@app.get("/endpoints")
def list_endpoints():
    return {
        "studio": STUDIO_NAME,
        "version": API_VERSION,
        "naming_rules": NAMING_RULES,
        "endpoints": ENDPOINTS,
    }


@app.get("/references/status")
def references_status():
    return {
        "studio": STUDIO_NAME,
        "references": get_all_references_status(),
    }


@app.post("/references/{matrix}")
async def upload_reference(
    background_tasks: BackgroundTasks,
    matrix: str,
    file: UploadFile = File(...),
):
    original_name = file.filename or "referencia.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
        raise HTTPException(status_code=400, detail="Formato inválido. Envie WAV, MP3, OGG ou FLAC.")

    tmp_path = os.path.join(ensure_temp_dir(), f"ref_up_{uuid.uuid4().hex[:8]}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        meta = save_reference_audio(tmp_path, matrix, original_name)
        background_tasks.add_task(cleanup_files, [tmp_path])
        return {"status": "ok", "message": f"Referência {meta['label']} salva com sucesso.", "reference": meta}
    except ValueError as e:
        cleanup_files([tmp_path])
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        cleanup_files([tmp_path])
        raise HTTPException(status_code=500, detail=f"Erro ao processar referência: {str(e)}")


@app.delete("/references/{matrix}")
def remove_reference(matrix: str):
    try:
        removed = delete_reference(matrix)
        return {
            "status": "ok",
            "removed": removed,
            "message": "Referência removida. O estúdio usará o preset PuigTec padrão." if removed else "Nenhuma referência salva.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/enhance")
async def enhance_route(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    matrix: str = Form(...),
    use_reference: bool = Form(True),
):
    if len(files) != 1:
        raise HTTPException(status_code=400, detail="A rota /enhance aceita exatamente 1 arquivo por vez.")

    upload = files[0]
    original_name = upload.filename or "audio.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (".wav", ".mp3"):
        raise HTTPException(status_code=400, detail="Formato inválido. Envie WAV ou MP3.")

    in_path = os.path.join(ensure_temp_dir(), f"in_{uuid.uuid4().hex[:8]}{ext}")
    temp_files = [in_path]

    try:
        with open(in_path, "wb") as f:
            f.write(await upload.read())

        out_path, out_filename, ref_applied = enhance_single_track(
            in_path, original_name, matrix, use_reference=use_reference
        )
        temp_files.append(out_path)
        background_tasks.add_task(cleanup_files, temp_files)
        return FileResponse(
            path=out_path,
            media_type="audio/wav",
            filename=out_filename,
            headers={
                "X-Matrix-Used": matrix.lower(),
                "X-Reference-Applied": str(ref_applied).lower(),
                "X-Output-Filename": out_filename,
            },
        )
    except ValueError as e:
        cleanup_files(temp_files)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        cleanup_files(temp_files)
        raise HTTPException(status_code=500, detail=f"Erro ao processar áudio: {str(e)}")


@app.post("/mixar-final")
async def mixar_final_route(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    playback: UploadFile | None = File(None),
    use_playback: bool = Form(False),
    track_types: str | None = Form(None),
    guitar_roles: str | None = Form(None),
    project_name: str | None = Form(None),
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Envie pelo menos 2 faixas tratadas (PRE-MIX+_) para mixar.")

    temp_dir = ensure_temp_dir()
    input_temp_files: list[str] = []
    track_paths: list[str] = []
    original_names: list[str] = []

    try:
        for idx, up in enumerate(files):
            orig = up.filename or f"track_{idx}.wav"
            ext = os.path.splitext(orig)[1].lower() or ".wav"
            tmp_in = os.path.join(temp_dir, f"mix_in_{idx}_{uuid.uuid4().hex[:8]}{ext}")
            with open(tmp_in, "wb") as f:
                f.write(await up.read())
            input_temp_files.append(tmp_in)
            track_paths.append(tmp_in)
            original_names.append(orig)

        pb_path = None
        if use_playback and playback is not None and playback.filename:
            pb_ext = os.path.splitext(playback.filename)[1].lower() or ".wav"
            pb_path = os.path.join(temp_dir, f"pb_{uuid.uuid4().hex[:8]}{pb_ext}")
            with open(pb_path, "wb") as f:
                f.write(await playback.read())
            input_temp_files.append(pb_path)

        out_path, out_filename, resolution_notes = mix_treated_tracks(
            track_paths=track_paths,
            original_filenames=original_names,
            track_types=_parse_list_field(track_types),
            guitar_roles=_parse_list_field(guitar_roles),
            playback_path=pb_path,
            use_playback=use_playback,
            project_name=project_name,
        )

        input_temp_files.append(out_path)
        background_tasks.add_task(cleanup_files, input_temp_files)
        return FileResponse(
            path=out_path,
            media_type="audio/wav",
            filename=out_filename,
            headers={
                "X-Output-Filename": out_filename,
                "X-Resolved-Types": " | ".join(resolution_notes),
            },
        )
    except ValueError as e:
        cleanup_files(input_temp_files)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        cleanup_files(input_temp_files)
        raise HTTPException(status_code=500, detail=f"Erro ao mixar faixas: {str(e)}")
