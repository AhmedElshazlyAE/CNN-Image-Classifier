from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['airplane', 'automobile', 'bird',
           'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck']

model = torch.load("models/cifar_pretrained_resnet_finetune.pth", map_location=device)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

@app.get("/")
def read_root():
    return {"message": "CIFAR-10 ResNet-18 API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "filename": file.filename,
        "predicted_class": classes[predicted.item()],
        "confidence": float(confidence.item()) * 100
    }