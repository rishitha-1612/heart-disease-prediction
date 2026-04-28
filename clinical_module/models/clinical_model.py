import joblib
import pandas as pd
import numpy as np

# ── Load model & scaler ─────────────────────────────
model = joblib.load("clinical_module/models/stack_model.pkl")
scaler = joblib.load("clinical_module/models/scaler.pkl")

# ── Label mapping ───────────────────────────────────
LABEL_MAP = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# ── Feature columns (must match training) ───────────
FEATURE_COLS = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "smoke", "alco", "active", "bmi", "hypertension",
    "pulse_pressure", "age_bmi_interaction", "bp_ratio",
    "bmi_age_ratio", "chol_gluc_product", "lifestyle_score",
    "cholesterol_2", "cholesterol_3", "gluc_2", "gluc_3"
]

# ── Build input features ────────────────────────────
def build_input(input_data):

    age = input_data["age"]
    height = input_data["height"]
    weight = input_data["weight"]
    ap_hi = input_data["ap_hi"]
    ap_lo = input_data["ap_lo"]
    cholesterol = input_data["cholesterol"]
    gluc = input_data["gluc"]

    bmi = weight / ((height / 100) ** 2)

    row = {
        "age": age,
        "gender": input_data["gender"],
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "smoke": input_data["smoke"],
        "alco": input_data["alco"],
        "active": input_data["active"],

        "bmi": bmi,
        "hypertension": int(ap_hi >= 140 or ap_lo >= 90),
        "pulse_pressure": ap_hi - ap_lo,
        "age_bmi_interaction": age * bmi,
        "bp_ratio": ap_hi / (ap_lo + 1),
        "bmi_age_ratio": bmi / (age + 1),
        "chol_gluc_product": cholesterol * gluc,
        "lifestyle_score": input_data["smoke"] + input_data["alco"] - input_data["active"],

        "cholesterol_2": int(cholesterol == 2),
        "cholesterol_3": int(cholesterol == 3),
        "gluc_2": int(gluc == 2),
        "gluc_3": int(gluc == 3),
    }

    df = pd.DataFrame([row]).reindex(columns=FEATURE_COLS, fill_value=0)
    return df


# ── FINAL PREDICT FUNCTION (Fusion uses this) ───────
def predict(input_data):

    df = build_input(input_data)
    scaled = scaler.transform(df)

    proba = model.predict_proba(scaled)[0]
    label = int(np.argmax(proba))

    return {
        "level": LABEL_MAP[label],
        "score": float(proba[label])
    }


if __name__ == "__main__":
    sample = {
        "age": 50,
        "gender": 2,
        "height": 170,
        "weight": 70,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    print(predict(sample))