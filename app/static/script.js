const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");
const chooseBtn = document.getElementById("choose-btn");
const previewImage = document.getElementById("preview-image");
const predictBtn = document.getElementById("predict-btn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");
const predictedClass = document.getElementById("predicted-class");
const confidence = document.getElementById("confidence");
const top3List = document.getElementById("top3-list");

let selectedFile = null;

chooseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
});

dropArea.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("drag-over");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("drag-over");
});

dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    dropArea.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];
    if (file) {
        handleFile(file);
    }
});

function handleFile(file) {
    clearMessages();

    if (!file.type.startsWith("image/")) {
        showError("Please upload a valid image file.");
        return;
    }

    selectedFile = file;

    const imageUrl = URL.createObjectURL(file);
    previewImage.src = imageUrl;

    dropArea.classList.add("has-image");
    predictBtn.disabled = false;
}

predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        showError("Please choose an image first.");
        return;
    }

    clearMessages();
    loading.classList.remove("hidden");
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        showResult(data);
    } catch (error) {
        showError(error.message);
    } finally {
        loading.classList.add("hidden");
        predictBtn.disabled = false;
    }
});

function showResult(data) {
    predictedClass.textContent = data.predicted_class;
    confidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;

    top3List.innerHTML = "";

    data.top_3.forEach((item) => {
        const percent = item.confidence * 100;

        const row = document.createElement("div");
        row.className = "top3-item";

        row.innerHTML = `
            <div class="top3-label">
                <strong>${item.class}</strong>
                <span>${percent.toFixed(2)}%</span>
            </div>
            <div class="bar">
                <div class="bar-fill" style="width: ${percent}%"></div>
            </div>
        `;

        top3List.appendChild(row);
    });

    result.classList.remove("hidden");
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function clearMessages() {
    result.classList.add("hidden");
    errorBox.classList.add("hidden");
}