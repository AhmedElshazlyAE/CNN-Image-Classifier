from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app

client = TestClient(app)

def make_test_image_bytes():
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_valid_image():
    image_bytes = make_test_image_bytes()

    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert "confidence" in data
    assert "top_3" in data
    assert len(data["top_3"]) == 3

def test_predict_invalid_file():
    response = client.post(
        "/predict",
        files={"file": ("bad.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400