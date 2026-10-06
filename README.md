# Histopathology Image Classification & Retrieval — Streamlit App

Interactive demo for **"Adaptive Multi-Scale Feature Fusion with Hybrid Representation Learning
for Histopathological Image Classification and Content-Based Retrieval"**
(manuscript ID computers-4369511).

A single model (ConvNeXt V2 + CBAM backbone → adaptive weighted fusion neck → hybrid head)
produces **one 18,496-dimensional descriptor** that is used at the same time for
tissue classification, content-based image retrieval, calibration and Grad-CAM explanation.
It was trained on **WSSS4LUAD** (lung adenocarcinoma, 4 classes: Normal, Stroma, Tumor, Tumor-Stroma).

## Pages

| Page | What it shows |
|------|---------------|
| **Retrieval Demo** | Upload an H&E patch (or pick a sample) → predicted class, confidence, Grad-CAM, top-6 similar cases |
| **Step-by-Step** | The retrieval pipeline one stage at a time: upload → preprocessing → backbone → fusion → 18,496-D descriptor → similarity search → result |
| **Overview** | Architecture diagram, model and dataset facts, key contributions |
| **Results** | All classification, retrieval and calibration metrics with figures |
| **Comparisons** | Baselines, ablation, 5-fold cross-validation, class balancing, cross-organ external validation |
| **Algorithms** | Plain-language pseudocode for the nine components |

## Model weights

The trained weights (`model_weights.pth`, 1.1 GB) are **not stored in this repository**
(GitHub's file limit is 100 MB). The app finds them in this order:

1. the path in the `MODEL_WEIGHTS` environment variable, if set;
2. `checkpoints/model_weights.pth`;
3. otherwise it downloads `model_weights.pth` from the Hugging Face model repo named in
   `HF_MODEL_REPO` (plus `HF_TOKEN` if that repo is private).

## Run locally

```bash
pip install -r requirements.txt
# put the weights at checkpoints/model_weights.pth, then:
streamlit run streamlit_app.py
```

Open <http://localhost:8501>. The app runs on CPU; no GPU is needed
(set `MODEL_DEVICE=cuda` to use one).

## Deploy on Streamlit Community Cloud

1. Upload `model_weights.pth` to a Hugging Face model repository.
2. On <https://share.streamlit.io>, create an app from this repository with
   main file `streamlit_app.py`.
3. Under **Advanced settings → Secrets**, add:

   ```toml
   HF_MODEL_REPO = "your-hf-username/your-model-repo"
   # HF_TOKEN = "hf_..."   # only if the Hugging Face repo is private
   ```

## Repository layout

```
streamlit_app.py      Streamlit front end (6 pages)
engine.py             model loading, classification, retrieval, Grad-CAM, step-by-step trace
presentation.py       tables and facts reported in the manuscript
src/models/           model architecture (backbone, neck, head)
gallery/              116 reference WSSS4LUAD patches + cached descriptor index
samples/              8 held-out sample queries
results/              metrics.json and all figures
static/style.css      shared stylesheet
```
