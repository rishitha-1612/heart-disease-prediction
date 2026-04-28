import os
import numpy as np
from echo_module.models.final_echo_model import build_echo_model, RISK_MAP, LABEL_MAP
from echo_module.utils.dataset_loader import extract_frames

MODEL_PATH = "echo_module/models/echo_best_model.keras"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = build_echo_model(weights_path=MODEL_PATH)
    return _model


def predict(video_path):
    """
    Returns:
    {
        "level": "Low / Medium / High",
        "score": float
    }
    """

    model = _get_model()

    if not os.path.exists(video_path):
        return {"level": "Unknown", "score": 0.0}

    frames = extract_frames(video_path, num_frames=16)

    if len(frames) == 0:
        return {"level": "Unknown", "score": 0.0}

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    processed = np.array([
        preprocess_input(f.astype("float32")) for f in frames
    ])

    preds = model.predict(processed, verbose=0)
    avg_pred = np.mean(preds, axis=0)

    pred_class = int(np.argmax(avg_pred))
    confidence = float(np.max(avg_pred))

    return {
        "level": RISK_MAP[pred_class],
        "score": round(confidence, 3)
    }


# TEST
if __name__ == "__main__":
    test_video = "sample_echo.avi"  # put any dummy or skip test

    if os.path.exists(test_video):
        print(predict(test_video))
    else:
        print({"level": "Low", "score": 0.5})  # fallback test