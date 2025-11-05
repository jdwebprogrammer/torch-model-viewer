
# Torch Model Viewer  
**Visualize PyTorch weights in 1D, 2D & 3D — instantly!**

A **2-file**, **zero-install** web dashboard to explore `.pth` / `.pt` models.  
No training, no setup — just drop in your model and **see every layer** in **colorful, interactive 3D**.
---
<img width="1368" height="734" alt="Screenshot_2025-11-05_10-42-56" src="https://github.com/user-attachments/assets/6adf5807-4aa9-49fd-b1b6-a8eda7eb5c02" />

<img width="1110" height="659" alt="Screenshot_2025-11-05_10-45-59" src="https://github.com/user-attachments/assets/680c9e58-e6bd-4586-b498-bf18513a21df" />

<img width="1110" height="659" alt="Screenshot_2025-11-05_10-50-56" src="https://github.com/user-attachments/assets/16324921-8f00-4c6b-914b-66ad4bcc9b9f" />
---

## Features

| Feature | Description |
|--------|-------------|
| **3D Point Cloud** | Red = low values, Blue = high values, **Size scales with magnitude** |
| **2D Heatmap** | Full-resolution layer view (auto-scaled) |
| **1D Line Chart** | Raw weight values (sampled) |
| **Histogram** | Distribution of values per layer |
| **Smart Indicators** | Overfit, dead weights, extreme values |
| **Dark Mode UI** | Clean, collapsible panels |

---

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/jdwebprogrammer/torch-model-viewer.git
cd torch-model-dashboard

# 2. Install (once)
pip install flask torch pillow numpy

# 3. Run!
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) → **Upload your `.pth` or `.pt` file** → Explore!

---

## How It Works

- **Backend (`app.py`)**: Loads model → extracts weights → computes stats, heatmaps, 3D layouts  
- **Frontend (`templates/index.html`)**: Three.js + Chart.js → interactive 3D orbit + responsive UI  
- **Smart Sampling**: Up to 10,000 points per layer → smooth performance even on huge models

---

## Why Use It?

- Spot **dead neurons** (all red, tiny)  
- Find **exploding gradients** (huge blue points)  
- Compare **layer patterns** across models  
- Debug **quantization**, **pruning**, or **training bugs**

---

## Project Structure

```
torch-model-dashboard/
├── app.py              Flask + Torch backend
├── templates/
│   └── index.html      Three.js + Chart.js frontend
└── README.md
```

---

## License

**MIT** — Free to use, modify, and share.

---
```
> Made with **love** for the PyTorch community  
> *“See your weights. Understand your model.”*
```
---


