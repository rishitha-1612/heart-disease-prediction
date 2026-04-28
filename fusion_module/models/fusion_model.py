class FusionModel:

    def level_to_num(self, level):
        mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3
        }
        return mapping[level]

    def num_to_level(self, score):
        if score < 1.5:
            return "Low"
        elif score < 2.5:
            return "Medium"
        else:
            return "High"

    def predict(self, echo, ecg, clinical):

        e = self.level_to_num(echo["level"])
        g = self.level_to_num(ecg["level"])
        c = self.level_to_num(clinical["level"])

        avg_score = (e + g + c) / 3

        final_level = self.num_to_level(avg_score)

        risk_percentage = avg_score / 3 * 100

        return {
            "final_level": final_level,
            "risk_percentage": round(risk_percentage, 2)
        }


if __name__ == "__main__":
    fusion = FusionModel()

    echo = {"level": "Medium", "score": 0.39}
    ecg = {"level": "High", "score": 0.66}
    clinical = {"level": "High", "score": 0.97}

    result = fusion.predict(echo, ecg, clinical)

    print(result)