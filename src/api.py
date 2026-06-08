from datetime import datetime
from pathlib import Path
import shutil
import json

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.graph import build_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = PROJECT_ROOT / "outputs" / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Image Processor on Steroids")

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
        },
    )


@app.post("/edit", response_class=HTMLResponse)
async def edit_image(
    request: Request,
    image: UploadFile = File(...),
    instruction: str = Form(...),
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / image.filename

    with input_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    agent = build_graph()

    result = agent.invoke(
        {
            "user_instruction": instruction,
            "input_image_path": str(input_path),
            "output_dir": str(run_dir),
            "logs": [],
        }
    )

    final_image_path = result.get("final_image_path")
    logs_md_path = run_dir / "logs.md"
    logs_json_path = run_dir / "logs.json"

    final_image_url = None
    if final_image_path:
        final_image_url = "/" + str(Path(final_image_path).relative_to(PROJECT_ROOT))

    logs_text = ""
    if logs_json_path.exists():
        logs_text = logs_json_path.read_text(encoding="utf-8")

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": {
                "status": "failed" if result.get("error") else "ok",
                "error": result.get("error"),
                "instruction": instruction,
                "final_image_url": final_image_url,
                "output_dir": str(run_dir),
                "logs_text": logs_text,
            },
        },
    )
