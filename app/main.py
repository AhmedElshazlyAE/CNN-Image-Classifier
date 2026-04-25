from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import torch

from app.model_utils import load_model, get_transform, CLASSES

app = FastAPI(title="CIFAR-10 ResNet-18 API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "resnet18_cifar10_state_dict.pth"

model, device = load_model(str(MODEL_PATH))
transform = get_transform()

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "app" / "static"),
    name="static"
)

templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health_check():
    return {"message": "CIFAR-10 ResNet-18 API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=1)

    top_3 = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        top_3.append({
            "class": CLASSES[idx.item()],
            "confidence": round(float(prob.item()), 4)
        })

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": top_3[0]["class"],
        "confidence": top_3[0]["confidence"],
        "top_3": top_3
    })