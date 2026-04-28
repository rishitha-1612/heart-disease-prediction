
from echo_module.echo_model import predict as echo_predict
from ecg_module.models.ecg_model import predict as ecg_predict
from clinical_module.models.clinical_model import predict as clinical_predict


class FusionModel:

    def level_to_num(self, level):
        mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3
        }
        if level not in mapping:
            return None  # treat Unknown as Medium
        
        return mapping[level]

    def num_to_level(self, score):
        if score < 1.5:
            return "Low"
        elif score < 2.5:
            return "Medium"
        else:
            return "High"

    def predict(self, echo_input, ecg_input, clinical_input):

        echo = echo_predict(echo_input)
        ecg = ecg_predict(ecg_input)
        clinical = clinical_predict(clinical_input)

        print("\n--- Module Outputs ---")
        print("Echo     :", echo)
        print("ECG      :", ecg)
        print("Clinical :", clinical)

        # Convert levels
        e = self.level_to_num(echo["level"])
        g = self.level_to_num(ecg["level"])
        c = self.level_to_num(clinical["level"])

        # Ignore Unknown
        values = []
        weights = []

        # 🎯 Define weights (you can tune later)
        W_ECHO = 0.3
        W_ECG = 0.3
        W_CLINICAL = 0.4

        if e is not None:
            values.append(e * W_ECHO * echo["score"])
            weights.append(W_ECHO)

        if g is not None:
            values.append(g * W_ECG * ecg["score"])
            weights.append(W_ECG)

        if c is not None:
            values.append(c * W_CLINICAL * clinical["score"])
            weights.append(W_CLINICAL)

        if len(values) == 0:
            return {"final_level": "Unknown", "risk_percentage": 0.0}

        # Weighted average
        final_score = sum(values) / sum(weights)

        final_level = self.num_to_level(final_score)
        risk_percentage = final_score / 3 * 100

        return {
            "final_level": final_level,
            "risk_percentage": round(risk_percentage, 2)
        }

# 🔻 TEST FULL SYSTEM
if __name__ == "__main__":

    fusion = FusionModel()

    print("\n===== HEART DISEASE PREDICTION SYSTEM =====\n")

    # 🔹 Clinical Input
    print("Enter Clinical Data:")
    age = int(input("Age: "))
    gender = int(input("Gender (1=Female, 2=Male): "))
    height = float(input("Height (cm): "))
    weight = float(input("Weight (kg): "))
    ap_hi = int(input("Systolic BP (ap_hi): "))
    ap_lo = int(input("Diastolic BP (ap_lo): "))
    cholesterol = int(input("Cholesterol (1/2/3): "))
    gluc = int(input("Glucose (1/2/3): "))
    smoke = int(input("Smoke (0/1): "))
    alco = int(input("Alcohol (0/1): "))
    active = int(input("Physical Activity (0/1): "))

    clinical_input = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }

    # 🔹 ECG Input (dummy for now)
    print("\nUsing sample ECG signal...")
    ecg_input = [[0.1] * 1000 for _ in range(12)]

    # 🔹 Echo Input (dummy for now)
    print("Using sample Echo video...")
    echo_input = "sample_echo.avi"

    # 🔹 Run system
    result = fusion.predict(echo_input, ecg_input, clinical_input)

    print("\n===== FINAL RESULT =====")
    print(result)































































# class FusionModel:

#     def level_to_num(self, level):
#         mapping = {
#             "Low": 1,
#             "Medium": 2,
#             "High": 3
#         }
#         return mapping[level]

#     def num_to_level(self, score):
#         if score < 1.5:
#             return "Low"
#         elif score < 2.5:
#             return "Medium"
#         else:
#             return "High"

#     def predict(self, echo, ecg, clinical):

#         e = self.level_to_num(echo["level"])
#         g = self.level_to_num(ecg["level"])
#         c = self.level_to_num(clinical["level"])

#         avg_score = (e + g + c) / 3

#         final_level = self.num_to_level(avg_score)

#         risk_percentage = avg_score / 3 * 100

#         return {
#             "final_level": final_level,
#             "risk_percentage": round(risk_percentage, 2)
#         }


# if __name__ == "__main__":
#     fusion = FusionModel()

#     echo = {"level": "Medium", "score": 0.39}
#     ecg = {"level": "High", "score": 0.66}
#     clinical = {"level": "High", "score": 0.97}

#     result = fusion.predict(echo, ecg, clinical)

#     print(result)