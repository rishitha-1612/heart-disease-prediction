class DummyFusion:
    """
    Simple rule-based fusion logic
    """

    def __init__(self):
        pass

    def predict(self, echo_result, ecg_result, clinical_result):
        """
        Inputs are binary (0 or 1)
        1 = disease / abnormal
        0 = normal
        """

        total = echo_result + ecg_result + clinical_result

        # Simple rule
        if total == 0:
            return "Low"
        elif total == 1:
            return "Medium"
        else:
            return "High"

if __name__ == "__main__":
    fusion = DummyFusion()

    result = fusion.predict(
        echo_result=1,
        ecg_result=0,
        clinical_result=1
    )

    print("Final Risk Level:", result)