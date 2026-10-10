import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "student_model.pkl"

data = {
    "Attendance": [
        85, 70, 95, 60, 90, 78, 92, 65, 88, 80,
        75, 98, 55, 82, 87, 68, 93, 72, 89, 77,
        91, 64, 84, 96, 73, 86, 79, 94, 67, 81
    ],
    "Study_Hours": [
        4, 2, 6, 2, 5, 3, 5, 2, 4, 3,
        3, 7, 1, 4, 5, 2, 6, 3, 5, 4,
        6, 2, 4, 7, 3, 5, 4, 6, 2, 4
    ],
    "Previous_Marks": [
        75, 55, 88, 45, 82, 65, 85, 50, 78, 68,
        62, 95, 40, 72, 80, 53, 90, 60, 83, 70,
        87, 48, 74, 96, 58, 81, 69, 92, 51, 76
    ]
}

df = pd.DataFrame(data)

# Synthetic demonstration target, not real student outcomes.
df["Final_Marks"] = (
    0.25 * df["Attendance"]
    + 2.5 * df["Study_Hours"]
    + 0.50 * df["Previous_Marks"]
).clip(0, 100)

X = df[["Attendance", "Study_Hours", "Previous_Marks"]]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Model saved at: {MODEL_PATH}")