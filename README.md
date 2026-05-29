# Automatic Agricultural Diseases Detection

An academic Machine Learning and Deep Learning project for automatic plant leaf disease detection.  
The project is inspired by the paper **"Machine Learning and Deep Learning Based Computational Techniques in Automatic Agricultural Diseases Detection: Methodologies, Applications, and Challenges"** and implements a complete workflow from dataset preparation to a web-based prediction interface.

## Project Overview

The application classifies tomato and potato leaf images into healthy or diseased classes. It compares classical Machine Learning models with Deep Learning models, then deploys the final TensorFlow/Keras model through a FastAPI backend and a Next.js frontend.

Supported classes:

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`
- `Tomato_Early_blight`
- `Tomato_Late_blight`
- `Tomato_healthy`

## Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Extraction
   ↓
Machine Learning Models
   ↓
Deep Learning Models
   ↓
Evaluation
   ↓
Comparison
   ↓
Explainability (Saliency Map)
   ↓
Frontend (Next.js)
   ↓
FastAPI Backend
   ↓
TensorFlow Model
   ↓
Prediction + Saliency Map
```

## Project Architecture

```text
plant-disease-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI routes and CORS configuration
│   │   └── ml_service.py        # Model loading, preprocessing, prediction, saliency map
│   ├── requirements.txt         # Backend Python dependencies
│   └── README.md
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Main Next.js user interface
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── package.json
│   └── package-lock.json
│
├── models/
│   ├── class_names.txt          # Class labels used by the backend
│   ├── efficientnet_model.h5    # Final TensorFlow/Keras model used at runtime
│   ├── confusion_matrix.png
│   ├── confusion_matrix_EfficientNet.png
│   └── custom_cnn_accuracy.png
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   ├── preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   ├── ml_classification.ipynb
│   ├── deep_learning_models.ipynb
│   ├── gradcam_visualization.ipynb
│   └── saliency_map.png
│
├── dataset/                     # Local dataset folder, not tracked by Git
├── rapport_overleaf.tex         # Academic LaTeX report
├── test_leaf.jpg                # Example image for testing
└── README.md
```

## Methodology

### 1. Dataset

The dataset contains tomato and potato leaf images. Each image belongs to one of the six supported classes. The local dataset used during development contains 6,652 images.

Class distribution used in the experiments:

| Class | Images |
|---|---:|
| Potato Early blight | 1000 |
| Potato Late blight | 1000 |
| Potato healthy | 152 |
| Tomato Early blight | 1000 |
| Tomato Late blight | 1909 |
| Tomato healthy | 1591 |

The `dataset/` directory is ignored by Git because datasets can be large. To reproduce training, place the dataset in:

```text
dataset/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
└── Tomato_healthy/
```

### 2. Preprocessing

Images are resized to `224x224` pixels.  
For classical Machine Learning, OpenCV is used to read and convert images.  
For Deep Learning, TensorFlow image pipelines and EfficientNet preprocessing are used.

### 3. Feature Extraction

Classical Machine Learning models are trained on handcrafted visual features:

- HOG features for shape and edge information
- RGB color histograms for color distribution
- PCA for dimensionality reduction

### 4. Machine Learning Models

The following models were trained and compared:

- K-Nearest Neighbors
- Support Vector Machine with RBF kernel
- Decision Tree
- Dense Artificial Neural Network

Best classical ML result:

| Model | Accuracy |
|---|---:|
| SVM RBF | 82.87% |

### 5. Deep Learning Models

Two Deep Learning approaches were implemented:

- Custom CNN
- EfficientNetB0 with transfer learning

Deep Learning results:

| Model | Validation Accuracy |
|---|---:|
| Custom CNN | 93.53% |
| EfficientNetB0 | 96.47% |
| EfficientNetB0 after fine-tuning | 95.86% |

### 6. Explainability

The backend generates a saliency map using TensorFlow gradients.  
The API returns:

- original preview image
- heatmap
- saliency overlay
- top predictions
- confidence score
- simple recommendation

## Backend API

The backend is built with **FastAPI**.

Main routes:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Check API status |
| `GET` | `/api/classes` | Return model name, input size and class labels |
| `POST` | `/api/predict` | Upload an image and return prediction results |

Prediction response includes:

- `filename`
- `model`
- `class_name`
- `label`
- `confidence`
- `is_healthy`
- `recommendation`
- `preview`
- `saliency.heatmap`
- `saliency.overlay`
- `top_predictions`

## Frontend

The frontend is built with **Next.js** and provides:

- image upload
- drag and drop interface
- API status indicator
- prediction result
- confidence score
- top predictions
- saliency and heatmap visualization

## Web Deployment

The project is now deployable as one web service:

- Next.js is exported as static files with `output: "export"`.
- FastAPI serves the exported frontend from `frontend/out`.
- The same public URL handles both the interface and `/api/*` routes.
- Docker builds the frontend, installs the backend, copies `models/efficientnet_model.h5`, then starts Uvicorn.

Deployment files:

- `Dockerfile`
- `.dockerignore`
- `render.yaml`

Important: `models/efficientnet_model.h5` must be pushed with the repository for cloud deployment. The file is about 49 MB, below GitHub's 100 MiB single-file limit.

### Deploy on Render

1. Push this repository to GitHub.
2. Make sure `models/efficientnet_model.h5` is included in the commit.
3. Open Render and create a new Blueprint from the GitHub repository.
4. Render will read `render.yaml`, build the Docker image and provide one public URL.

After deployment:

```text
https://your-service.onrender.com
https://your-service.onrender.com/docs
https://your-service.onrender.com/api/health
```

## Installation and Running

### Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- Git

### 1. Clone the Repository

```powershell
git clone https://github.com/MiloudiAhmed/Automatic-Agricultural-Diseases-Detection.git
cd Automatic-Agricultural-Diseases-Detection
```

### 2. Create and Activate a Python Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Backend Dependencies

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the FastAPI Backend

```powershell
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Install Frontend Dependencies

Open a second terminal:

```powershell
cd frontend
npm install
```

### 6. Run the Next.js Frontend

```powershell
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

## Optional Environment Variables

The backend loads the default model and class labels from:

```text
models/efficientnet_model.h5
models/class_names.txt
```

For deployment, `models/efficientnet_model.h5` is tracked by Git so Docker-based hosts can build the complete app. To run inference locally, keep the trained EfficientNet model in:

```text
models/efficientnet_model.h5
```

You can also train or regenerate the model from the notebooks, then copy the exported `.h5` file to the `models/` directory.

You can override these paths:

```powershell
$env:PLANT_MODEL_PATH="C:\path\to\model.h5"
$env:PLANT_CLASS_NAMES_PATH="C:\path\to\class_names.txt"
```

## Test the API

```powershell
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Notes About Large Files

The following generated files are intentionally not tracked by Git:

- `dataset/`
- `models/X_features.npy`
- `models/y_labels.npy`
- `models/custom_cnn_model.h5`
- `notebooks/efficientnet_functional.keras`
- virtual environments
- frontend build outputs
- `node_modules/`

The backend expects `models/efficientnet_model.h5` at runtime. Keep the file locally or provide a custom path through `PLANT_MODEL_PATH`.

## Academic Report

The LaTeX report is available in:

```text
rapport_overleaf.tex
```

It contains the full academic structure: title page, summary, workflow, methodology, evaluation, comparison, explainability, backend, frontend and execution commands.

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- FastAPI
- Uvicorn
- Pillow
- NumPy
- Next.js
- React
- TypeScript
- Lucide React

## Author

**Ahmed Miloudi**
