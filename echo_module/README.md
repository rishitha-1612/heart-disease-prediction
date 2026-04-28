# Echo Module — Heart Disease Prediction
 
**Model:** MobileNetV2 (Transfer Learning)  
**Dataset:** EchoNet-Dynamic (10,030 echocardiography videos)  
**Accuracy:** 70.8% (validation set)

---

## Folder Structure

```
echo_module/
├── models/
│   └── final_echo_model.py     ← MobileNetV2 architecture
├── utils/
│   └── dataset_loader.py       ← Frame extractor + data generator
├── echo_module.py              ← Main prediction function (for fusion)
└── README.md
```

---

## How It Works

1. Loads echocardiography video (.avi)
2. Extracts 16 evenly spaced frames
3. Runs each frame through MobileNetV2
4. Averages predictions across all frames
5. Maps to risk level for fusion module

---

## EF-Based Classification

| EF Range | Class | Risk Output |
|---|---|---|
| EF > 55% | Normal | Low |
| EF 40–55% | Mild | Low |
| EF 30–40% | Moderate | Medium |
| EF < 30% | Severe | High |

---

## For Sayeed (Fusion Module)

```python
from echo_module import get_echo_prediction

result = get_echo_prediction("path/to/video.avi")
print(result)
# {"level": "High", "score": 0.83}
```

---

## Setup

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn
```

Update `MODEL_PATH` in `echo_module.py` to point to `echo_best_model.keras`.

---

## Output Format

```json
{
    "level": "Low / Medium / High",
    "score": 0.0 to 1.0
}
```
