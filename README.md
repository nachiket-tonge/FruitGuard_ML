# Fruit Disease Detection — Backend API

Serves the trained **EfficientNet-B3** fruit disease classification model via a REST API built with FastAPI.

---

## Folder Structure

```
backend/
├── app.py              ← FastAPI server (routes + CORS)
├── model.py            ← Model loading + inference logic
├── severity.py         ← HSV-based disease severity analysis
├── treatments.py       ← Severity-tiered treatment advice lookup
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← For cloud / container deployment
└── model_output/
    ├── fruit_disease_efficientb3.pth      ← Main model (use this)
    ├── fruit_disease_efficientb3_scripted.pt
    ├── best_model.pth
    ├── classification_report.csv
    ├── confusion_matrix.png
    ├── per_class_f1.png
    └── sample_images.png
```

---

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
```

---

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
```

---

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
