import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from app.model import LABEL_NAMES, load_model
from app.preprocessing import ANATOM_CATEGORIES, preprocess_image, preprocess_metadata
from app.schemas import PredictionResponse

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "models/best_model_improved.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_state["model"] = load_model(CHECKPOINT_PATH, DEVICE)
    yield
    ml_state.clear()


app = FastAPI(title="Melanoma Classifier Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@torch.no_grad()
def predict_with_tta(model, image_tensor, metadata_tensor, device):
    """4-view TTA (gốc + flip ngang/dọc/cả hai) - giống evaluate_tta trong 07_train_improved.ipynb."""
    image_tensor = image_tensor.to(device)
    metadata_tensor = metadata_tensor.to(device)

    views = [
        image_tensor,
        torch.flip(image_tensor, dims=[3]),
        torch.flip(image_tensor, dims=[2]),
        torch.flip(image_tensor, dims=[2, 3]),
    ]
    probs = [torch.sigmoid(model(v, metadata_tensor)) for v in views]
    return torch.stack(probs).mean(dim=0).item()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"anatom_categories": ANATOM_CATEGORIES}
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    anatom_site: str = Form(...),
):
    if not (0 <= age <= 120):
        raise HTTPException(status_code=422, detail="age phải trong khoảng 0-120")

    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="File tải lên không phải ảnh hợp lệ")

    try:
        metadata_tensor = preprocess_metadata(age, sex, anatom_site)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    image_tensor = preprocess_image(pil_image)

    model = ml_state["model"]
    prob_melanoma = predict_with_tta(model, image_tensor, metadata_tensor, DEVICE)

    label = LABEL_NAMES[1] if prob_melanoma >= 0.5 else LABEL_NAMES[0]
    confidence = prob_melanoma if prob_melanoma >= 0.5 else 1 - prob_melanoma

    return PredictionResponse(
        label=label,
        probability_melanoma=round(prob_melanoma, 4),
        confidence=round(confidence, 4),
    )
