# 🍎 FruitGuard ML

The machine-learning service for **FruitGuard**, an AI-powered fruit health analysis platform.

FruitGuard ML is a Python/FastAPI service responsible for image-based fruit disease analysis. The service receives a fruit image, runs the trained deep-learning model, and returns prediction information that the FruitGuard frontend can use to present a clear diagnosis and severity assessment.

The service is designed to work alongside the FruitGuard Next.js frontend and Spring Boot backend.

---

## 🌱 What is FruitGuard?

FruitGuard is an AI-assisted fruit health platform designed to help users understand possible diseases affecting fruit crops.

The overall system is divided into separate services:

```text
                         FruitGuard
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       Next.js Frontend   Spring Boot    ML Service
                          Backend        FastAPI
              │              │              │
              │              │              │
              └──────────────┼──────────────┘
                             │
                       Fruit Analysis
```

### Current supported fruits

The current FruitGuard application supports:

- 🥑 Avocado
- 🍈 Guava
- 🥭 Mango
- 🍊 Orange
- ❤️ Pomegranate

The ML model should be used with images belonging to the supported fruit/disease classes for which it was trained.

---

# ✨ Features

- 🧠 Deep-learning based image classification
- 🖼️ Image upload through FastAPI
- ⚡ FastAPI inference API
- 🔥 PyTorch-based model inference
- 🏗️ ResNet50-based architecture
- 📦 Transfer learning using pretrained weights
- 🔍 Grad-CAM explainability
- 📍 Visual highlighting of model attention
- 📊 Disease prediction and confidence information
- 🌱 Designed specifically for FruitGuard's supported fruit classes
- 🐍 Python virtual-environment based setup
- 💻 CPU-compatible inference environment

---

# 🧠 Machine Learning Approach

FruitGuard ML uses a convolutional neural-network approach based on **ResNet50**.

The model uses transfer learning rather than training an entire deep neural network from scratch.

Conceptually:

```text
Input Fruit Image
       │
       ▼
Image Preprocessing
       │
       ▼
ResNet50
       │
       ├── Feature Extraction
       │
       ├── Deep Convolutional Layers
       │
       └── Classification Head
                │
                ▼
        Disease Prediction
                │
                ├── Predicted class
                ├── Confidence
                └── Additional analysis
```

The project uses TensorFlow/Keras in the model-training workflow and PyTorch/Torchvision/Timm in the current inference environment.

---

# 📊 Dataset

The FruitGuard model was trained using a combined fruit X-ray/image dataset containing approximately:

```text
10,580 images
```

The dataset was organized into training, validation, and testing portions.

The exact dataset structure and class distribution should be kept consistent with the model/preprocessing configuration used during training.

---

# 🏋️ Model Training

The training pipeline uses image preprocessing and transfer learning.

Important training characteristics include:

```text
Image size:       224 × 224
Model:            ResNet50
Batch size:       16
Training mode:    Transfer learning / fine-tuning
Framework:        TensorFlow / Keras
```

The model's final convolutional layer used for Grad-CAM analysis is:

```text
conv5_block3_3_conv
```

This layer is used to visualize the regions that contribute strongly to the model's prediction.

---

# 📈 Model Performance

The trained model reached approximately:

```text
Training Accuracy:   97.93%
Validation Accuracy: 91.13%
Test Accuracy:       96.80%
```

Corresponding recorded losses include:

```text
Training Loss: 0.0589
Validation Loss: 0.2478
Test Loss: 0.0854
```

These values describe the recorded model evaluation during development and should not be interpreted as a guarantee of real-world agricultural diagnosis accuracy.

---

# 🔥 Grad-CAM Explainability

FruitGuard uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to make the model's prediction easier to interpret visually.

Instead of returning only:

```text
Prediction: Disease X
```

the system can generate a heatmap showing the image regions that contributed most strongly to the prediction.

Conceptually:

```text
Original Image
      │
      ▼
   ResNet50
      │
      ▼
Predicted Disease
      │
      ▼
Gradient Calculation
      │
      ▼
Grad-CAM Heatmap
      │
      ▼
Highlighted Image
```

This is useful because the user can see whether the model is focusing on a visually relevant region of the fruit.

### Grad-CAM components

The project includes a Grad-CAM implementation that:

1. Loads the trained model.
2. Identifies the target convolutional layer.
3. Calculates gradients for the predicted class.
4. Generates a class activation heatmap.
5. Overlays the heatmap on the original image.
6. Can derive a bounding region around the activated area.

The configured final convolutional layer is:

```text
conv5_block3_3_conv
```

---

<<<<<<< HEAD
## Running Locally

### 1. Install dependencies

```bash
# Inside the backend/ folder
pip install -r requirements.txt
```

> If you have a GPU, PyTorch will automatically use it.
> For CPU-only installation:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install fastapi uvicorn pillow python-multipart
> ```

### 2. Start the server

```bash
# Inside the backend/ folder
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test it

Open your browser at:
- **Swagger UI (interactive docs):** http://localhost:8000/docs
- **Health check:**                  http://localhost:8000/
- **All class names:**               http://localhost:8000/classes

Or test with curl:
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/your/fruit_image.jpg"
=======
# 🏗️ ML Service Architecture

The runtime architecture is approximately:

```text
                   Client / Frontend
                           │
                           │ multipart/form-data
                           ▼
                    ┌──────────────┐
                    │   FastAPI    │
                    │   app.py     │
                    └──────┬───────┘
                           │
                           ▼
                    Image Validation
                           │
                           ▼
                    Image Preprocessing
                           │
                           ▼
                    ML Model Inference
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
             Prediction         Explainability
                  │                 │
                  │              Grad-CAM
                  │                 │
                  └────────┬────────┘
                           │
                           ▼
                     JSON Response
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | ML service language |
| FastAPI | HTTP inference API |
| Uvicorn | ASGI application server |
| PyTorch | Current inference runtime |
| Torchvision | Computer-vision utilities |
| Timm | PyTorch model utilities |
| Pillow | Image processing |
| OpenCV | Image processing and visualization |
| python-multipart | Multipart image uploads |
| NumPy | Numerical processing |
| TensorFlow / Keras | Model training workflow |
| ResNet50 | Deep-learning architecture |
| Grad-CAM | Model explainability |

---

# 📂 Project Structure

A typical FruitGuard ML project is organized around the FastAPI application, model files, preprocessing code, and explainability utilities.

```text
FruitGuard_ML/
│
├── venv/
│
├── app.py
│
├── models/
│   └── trained model files
│
├── gradcam.py
│
├── requirements.txt
│
└── README.md
```

Additional dataset, preprocessing, training, or utility files may exist depending on the development version of the project.

### Important directories/files

#### `app.py`

The FastAPI application entry point.

It exposes the ML service through HTTP endpoints and handles uploaded images/inference requests.

#### `models/`

Contains the trained model artifacts used for inference.

The model directory can be relatively large because deep-learning model weights are substantially larger than normal application source files.

#### `gradcam.py`

Contains the Grad-CAM/explainability functionality used to generate model attention visualizations.

#### `requirements.txt`

Defines the Python dependencies required by the ML service.

---

# 🐍 Python Environment

The project uses a Python virtual environment:

```text
venv/
```

The virtual environment should **not** be committed to Git.

Create a new environment with:

```bash
python3 -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```cmd
venv\Scripts\activate
```

---

# 📦 Dependencies

The current requirements are:

```txt
fastapi>=0.110.0
uvicorn[standard]>=0.29.0

torch==2.3.1
torchvision==0.18.1

timm>=0.9.0
pillow>=10.0.0
python-multipart>=0.0.9
opencv-python-headless>=4.9.0
```

The PyTorch versions are intentionally pinned:

```text
torch       2.3.1
torchvision 0.18.1
```

This makes the environment more predictable than using unrestricted `>=` versions for PyTorch.

---

# 💻 CPU Installation

The current development environment uses the CPU builds of PyTorch.

The installed versions are:

```text
torch       2.3.1+cpu
torchvision 0.18.1+cpu
```

This is intentional for the current development environment.

If you do not have an NVIDIA/CUDA environment, install the CPU wheels directly:

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
pip install timm pillow python-multipart opencv-python-headless
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_FRUITGUARD_ML_REPOSITORY_URL>
```

Then:

```bash
cd FruitGuard_ML
```

---

## 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 4. Install PyTorch CPU

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

If your requirements file does not include a CPU-specific PyTorch index, install the PyTorch CPU packages separately as shown above before running the requirements installation.

---

# ✅ Verify Installation

Run:

```bash
python -c "import fastapi, uvicorn, torch, torchvision, timm, PIL, cv2; print('ALL IMPORTS OK'); print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected CPU environment:

```text
ALL IMPORTS OK
Torch: 2.3.1+cpu
Torchvision: 0.18.1+cpu
CUDA available: False
```

`CUDA available: False` is expected for the CPU configuration.

---

# ▶️ Running the ML Service

Start the FastAPI application with:

```bash
uvicorn app:app --reload
```

The development server should start at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

You can also access the OpenAPI JSON at:

```text
http://127.0.0.1:8000/openapi.json
```

The exact API operations available in the current implementation should be inspected through the generated Swagger documentation.

---

# 🔌 API Usage

The FastAPI service accepts image-analysis requests through its defined API endpoints.

The general request flow is:

```text
POST request
     │
     ▼
Multipart image upload
     │
     ▼
FastAPI
     │
     ▼
Image preprocessing
     │
     ▼
Model inference
     │
     ▼
Prediction
     │
     ▼
Optional Grad-CAM processing
     │
     ▼
JSON response
```

Use:

```text
/docs
```

to see the exact endpoint paths, parameters, request format, and response schema exposed by the current `app.py`.

---

# 🖼️ Image Processing

Images are processed before being passed to the model.

The typical pipeline is:

```text
Uploaded Image
      │
      ▼
Read Image
      │
      ▼
Resize
224 × 224
      │
      ▼
Tensor Conversion
      │
      ▼
Normalization / Preprocessing
      │
      ▼
Model
```

The preprocessing used during inference must remain consistent with the preprocessing used during training.

Changing normalization, image size, color handling, or other preprocessing steps can significantly affect predictions.

---

# 🔍 Prediction Flow

At inference time:

```text
1. User uploads image
             │
             ▼
2. FastAPI receives image
             │
             ▼
3. Image is decoded
             │
             ▼
4. Image is resized/preprocessed
             │
             ▼
5. Model performs inference
             │
             ▼
6. Predicted class is obtained
             │
             ▼
7. Confidence/result information is prepared
             │
             ▼
8. Response returned to frontend
>>>>>>> 7778c546470f52a4a5d5afa481ecf6817ac8adc8
```

---

<<<<<<< HEAD
## API Reference

### `GET /`
Health check. Returns server status and model info.

### `GET /classes`
Returns all 21 class names the model recognises, grouped by fruit.

### `POST /predict`
Upload a fruit image and get a disease prediction.

**Request:**
```
Content-Type: multipart/form-data
Body field:   file  (image/jpeg | image/png | image/webp | image/bmp)
```

**Response:**
```json
{
  "result":            "Mango Anthracnose",
  "severity":          "Moderate",
  "diseased_area_pct": 34.7,
  "treatment":         "Apply copper-based fungicide every 10–14 days...",
  "highlighted_image": "<base64-encoded PNG string>"
}
```

| Field | Type | Description |
|---|---|---|
| `result` | string | Human-readable predicted class, e.g. `"Mango Anthracnose"` |
| `severity` | string | `"Healthy"` \| `"Mild"` \| `"Moderate"` \| `"Severe"` |
| `diseased_area_pct` | float | Estimated % of image pixels identified as diseased tissue |
| `treatment` | string | Severity-specific treatment or care advice |
| `highlighted_image` | string | Base64-encoded PNG with diseased regions highlighted in red |

To display the highlighted image in any frontend:
```html
<!-- HTML -->
<img src="data:image/png;base64,{{ result.highlighted_image }}">
```
```javascript
// JavaScript
img.src = "data:image/png;base64," + result.highlighted_image;
```

---

## Supported Classes (21 total)

| Fruit | Classes |
|---|---|
| Avocado | Anthracnose, Healthy, Scab |
| Guava | Anthracnose, Healthy, Phytophthora, Scab, Styler_and_Root |
| Mango | Alternaria, Anthracnose, Healthy, Scab, Stem_end_Rot |
| Orange | Citrus_Canker, Citrus_Melanose, Healthy |
| Pomegranate | Alternaria, Anthracnose, Bacterial_Blight, Cercospora, Healthy |

---

## Deploying to the Cloud

### Option A: Render / Railway (easiest — free tier available)
1. Push this `backend/` folder to a GitHub repository
2. Create a new **Web Service** on [Render](https://render.com) or [Railway](https://railway.app)
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Done — you'll get a public URL like `https://your-app.onrender.com`

### Option B: Docker (any VPS / cloud)
```bash
# Build the image
docker build -t fruit-disease-api .

# Run it
docker run -p 8000:8000 fruit-disease-api
```

### CORS for Production
In `app.py`, update the CORS origin to your website's domain:
```python
allow_origins=["https://yourwebsite.com"]
=======
# 🔥 Explainability Flow

When Grad-CAM is requested/used:

```text
Image
 │
 ▼
Model
 │
 ▼
Prediction
 │
 ▼
Target class
 │
 ▼
Gradients
 │
 ▼
Target convolutional layer
 │
 ▼
Grad-CAM heatmap
 │
 ▼
Overlay on original image
 │
 ▼
Highlighted result
```

This allows FruitGuard to provide more than a simple classification label.

---

# ⚠️ Important ML Considerations

## Model confidence is not certainty

A high-confidence model prediction does not guarantee that the prediction is correct.

The model should therefore be treated as a decision-support system rather than a replacement for a qualified agricultural expert.

## Image quality matters

Poor images can reduce model performance.

For best results:

- Use good lighting.
- Keep the fruit in focus.
- Avoid extreme blur.
- Keep the diseased region visible.
- Avoid unnecessary background clutter.
- Use images similar to the training data.

## Supported fruits matter

The current FruitGuard application supports:

```text
Avocado
Guava
Mango
Orange
Pomegranate
```

Images outside the supported domain may produce unreliable predictions.

---

# 🧪 Development and Debugging

Start the service with:

```bash
uvicorn app:app --reload
```

The `--reload` flag automatically restarts the development server when Python source files change.

For a normal non-reloading run:

```bash
uvicorn app:app
>>>>>>> 7778c546470f52a4a5d5afa481ecf6817ac8adc8
```

---

<<<<<<< HEAD
## For the Website / App Developer

No API key required. No authentication. Just send the image.

The API accepts `multipart/form-data` with a single field named `file`.

```javascript
// Upload an image and receive a full prediction
const formData = new FormData();
formData.append("file", imageFile);   // imageFile = <input type="file"> value

const response = await fetch("https://YOUR-API-URL/predict", {
  method: "POST",
  body: formData,
  // Do NOT set Content-Type — the browser sets the multipart boundary automatically
});

if (!response.ok) {
  const err = await response.json();
  console.error(err.detail);  // e.g. "Unsupported file type: 'image/gif'"
  return;
}

const result = await response.json();
console.log(result.result);            // "Mango Anthracnose"
console.log(result.severity);          // "Moderate"
console.log(result.diseased_area_pct); // 34.7
console.log(result.treatment);         // "Apply copper-based fungicide..."

// Display the highlighted image
document.getElementById("highlight").src =
  "data:image/png;base64," + result.highlighted_image;
```

### Supported image formats
`image/jpeg`, `image/png`, `image/webp`, `image/bmp`

### Error responses
| HTTP Status | Meaning |
|---|---|
| `415` | Unsupported file type |
| `400` | Could not decode/read the uploaded file |
| `500` | Inference failed (model error) |

Errors return `{ "detail": "<description>" }`.
=======
# 🐛 Troubleshooting

## `ModuleNotFoundError: No module named 'fastapi'`

Make sure the virtual environment is active:

```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

---

## `ModuleNotFoundError` for torch/timm/PIL/cv2

Verify:

```bash
pip list
```

Then test:

```bash
python -c "import torch, torchvision, timm, PIL, cv2; print('ML imports OK')"
```

Install the missing dependency if necessary.

---

## PyTorch tries to download huge CUDA packages

For the current CPU development environment, install:

```bash
pip uninstall -y torch torchvision torchaudio
```

Then:

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

```text
2.3.1+cpu
False
```

---

## SSL / network errors while installing PyTorch

Large PyTorch packages can fail during download because of network interruptions.

If installation fails partway through:

1. Activate the correct virtual environment.
2. Remove incomplete PyTorch packages if necessary.
3. Retry the CPU wheel installation.
4. Check your internet connection.
5. Avoid switching randomly between CUDA and CPU builds.

Recommended CPU installation:

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

---

# 📦 What Should NOT Be Committed

Do not commit the virtual environment:

```text
venv/
```

Also avoid committing:

```text
__pycache__/
*.pyc
.env
model caches
large generated files
temporary uploads
logs
```

A suitable `.gitignore` should include at least:

```gitignore
# Virtual environment
venv/
.venv/

# Python
__pycache__/
*.py[cod]
*$py.class

# Environment variables
.env
.env.*

# IDE
.idea/
.vscode/

# Logs
*.log

# OS
.DS_Store

# Generated files
uploads/
outputs/

# Model/cache files
.cache/
```

Model weights should only be stored in Git if they are intentionally part of the repository and the repository size/licensing constraints allow it. For larger models, use an appropriate model/artifact-storage solution.

---

# 🔗 Integration With FruitGuard

The ML service is one component of the larger FruitGuard system.

```text
┌──────────────────────────┐
│    FruitGuard Frontend   │
│      Next.js / React     │
└────────────┬─────────────┘
             │
             │ Image analysis request
             ▼
┌──────────────────────────┐
│     FruitGuard ML        │
│       FastAPI            │
│                          │
│  Image → Model → Result  │
└────────────┬─────────────┘
             │
             │ Prediction
             ▼
┌──────────────────────────┐
│    FruitGuard Frontend   │
│                          │
│ Result + Severity +      │
│ Treatment workflow       │
└────────────┬─────────────┘
             │
             │ Authenticated history
             ▼
┌──────────────────────────┐
│   FruitGuard Backend     │
│      Spring Boot         │
│                          │
│ Users + Detection       │
│ History + JWT           │
└──────────────────────────┘
```

The ML service is intentionally separated from the Spring Boot backend so that model inference can evolve independently from user management and application persistence.

---

# 🧠 Why a Separate ML Service?

Separating ML inference from the main backend provides several advantages:

### Independent environments

Python ML dependencies do not have to be installed in the Java backend.

### Independent deployment

The ML service can be deployed separately from the Spring Boot API.

### Easier model iteration

New model versions can be tested without restructuring the main application backend.

### Clear responsibilities

```text
Spring Boot
    │
    ├── Users
    ├── Authentication
    ├── History
    └── Application data

FastAPI
    │
    ├── Image processing
    ├── Model inference
    ├── Prediction
    └── Explainability
```

---

# 🚀 Future Improvements

Possible improvements include:

- Add stronger input validation
- Add model version information to responses
- Add confidence calibration
- Improve out-of-domain image detection
- Add more fruit/disease classes
- Improve Grad-CAM localization
- Add batch inference
- Add GPU deployment configuration
- Add model health checks
- Add inference timing metrics
- Add structured logging
- Add automated model evaluation
- Add model versioning
- Add Docker support
- Add CI/CD
- Add automated API tests
- Add production inference monitoring
- Add model artifact storage

---

# 📌 Current Status

The FruitGuard ML service currently provides the machine-learning component of the FruitGuard platform.

Current development environment:

```text
Python
FastAPI
Uvicorn
PyTorch 2.3.1+cpu
Torchvision 0.18.1+cpu
Timm
Pillow
OpenCV
python-multipart
```

The trained FruitGuard model is based on ResNet50 and the project includes Grad-CAM explainability.

The current supported FruitGuard fruit categories are:

```text
Avocado
Guava
Mango
Orange
Pomegranate
```

---

# 👨‍💻 Development Notes

This service is intended to be run independently from the frontend and Spring Boot backend during development.

Recommended local setup:

```text
Terminal 1
──────────
Spring Boot Backend
localhost:8080


Terminal 2
──────────
FruitGuard ML
localhost:8000


Terminal 3
──────────
Next.js Frontend
localhost:3000
```

This makes it easier to identify which service is responsible for a problem.

---

# 📄 License

Add the project's selected license here before distributing the repository publicly.

---

# 🙌 FruitGuard

FruitGuard brings together:

```text
Computer Vision
       +
Deep Learning
       +
Explainable AI
       +
FastAPI
       +
Spring Boot
       +
Next.js
```

The goal is to turn a fruit image into an understandable starting point for identifying potential disease problems and deciding what action to investigate next.
>>>>>>> 7778c546470f52a4a5d5afa481ecf6817ac8adc8
