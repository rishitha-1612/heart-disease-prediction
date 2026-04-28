"""
Echo Module — Main Integration File
Model   : MobileNetV2 (Transfer Learning)
Dataset : EchoNet-Dynamic (10,030 videos)

FOR SAYEED (Fusion Module):
from echo_module import get_echo_prediction

result = get_echo_prediction("path/to/video.avi")
print(result)
# {"level": "High", "score": 0.83}
"""

import os
import numpy as np
from models.final_echo_model import build_echo_model, RISK_MAP, LABEL_MAP
from utils.dataset_loader import extract_frames

# ── Update this path to where model is saved ──────────────────────────────
MODEL_PATH = "echo_best_model.keras"

# Global model instance (loaded once, reused for all predictions)
_model = None


def _get_model():
    """Load model once and cache it."""
    global _model
    if _model is None:
        _model = build_echo_model(weights_path=MODEL_PATH)
    return _model


def get_echo_prediction(video_path):
    """
    ══════════════════════════════════════════════════
    MAIN FUNCTION — called by Fusion Module (Sayeed)
    ══════════════════════════════════════════════════

    Takes a raw echocardiography video and returns
    a structured risk prediction ready for fusion.

    Parameters
    ----------
    video_path : str
        Path to .avi echocardiography video file.

    Returns
    -------
    dict
        {
            "level" : "Low" / "Medium" / "High",
            "score" : float (0.0 to 1.0)
        }

    Example
    -------
    >>> result = get_echo_prediction("0X1002E8FBACD08477.avi")
    >>> print(result)
    {"level": "Low", "score": 0.72}
    """
    model = _get_model()

    # Validate video exists
    if not os.path.exists(video_path):
        print(f"⚠️  Video not found: {video_path}")
        return {"level": "Unknown", "score": 0.0}

    # Extract frames
    frames = extract_frames(video_path, num_frames=16)
    if len(frames) == 0:
        print(f"⚠️  Could not extract frames: {video_path}")
        return {"level": "Unknown", "score": 0.0}

    # Preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    processed = np.array([
        preprocess_input(f.astype("float32")) for f in frames
    ])

    # Predict — average across all frames (video-level)
    preds      = model.predict(processed, verbose=0)
    avg_pred   = np.mean(preds, axis=0)
    pred_class = int(np.argmax(avg_pred))
    confidence = float(np.max(avg_pred))

    return {
        "level": RISK_MAP[pred_class],
        "score": round(confidence, 3)
    }


def get_echo_batch(video_paths):
    """
    Run predictions on multiple videos.
    More efficient than calling get_echo_prediction() in a loop
    because model is loaded only once.

    Parameters
    ----------
    video_paths : list of str

    Returns
    -------
    list of dict — [{"level": ..., "score": ...}, ...]
    """
    results = []
    for path in video_paths:
        result = get_echo_prediction(path)
        results.append(result)
        print(f"  {os.path.basename(path):30s} → {result}")
    return results


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("    ECHO MODULE — INTEGRATION TEST")
    print("=" * 55)

    TEST_VIDEO = "sample_echo.avi"   # replace with real path

    if os.path.exists(TEST_VIDEO):
        output = get_echo_prediction(TEST_VIDEO)
        print(f"\n  Video  : {TEST_VIDEO}")
        print(f"  Level  : {output['level']}")
        print(f"  Score  : {output['score']}")
        print(f"\n  ✅ Ready for fusion module")
    else:
        print(f"\n  ⚠️  Update TEST_VIDEO path to run test")

    print("=" * 55)
