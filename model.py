import pickle
from pathlib import Path
import pandas as pd

MODEL_PATH = Path(__file__).parent / "student_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def predict_performance(attendance, study_hours, previous_marks):
    model = load_model()

    input_data = pd.DataFrame(
        [[attendance, study_hours, previous_marks]],
        columns=["Attendance", "Study_Hours", "Previous_Marks"]
    )

    prediction = model.predict(input_data)[0]
    return round(max(0, min(100, float(prediction))), 2)


def get_performance_category(marks):
    if marks >= 80:
        return "Excellent"
    elif marks >= 60:
        return "Good"
    elif marks >= 40:
        return "Average"
    else:
        return "Needs Improvement"