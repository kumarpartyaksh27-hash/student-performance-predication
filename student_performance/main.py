import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Student Performance Dataset
data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 7, 8, 3, 6],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 92, 95, 72, 78, 88, 91, 68, 82],
    "Past_Score": [50, 55, 60, 65, 70, 75, 80, 85, 90, 62, 68, 77, 83, 58, 72],
    "Final_Score": [52, 58, 63, 68, 74, 80, 86, 92, 98, 65, 71, 81, 87, 60, 76]
}

df = pd.DataFrame(data)

print("\n===== STUDENT PERFORMANCE PREDICTION =====")
print("\nDataset:")
print(df)

# Features and target
X = df[["Hours_Studied", "Attendance", "Past_Score"]]
y = df["Final_Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# New student prediction
hours = 7
attendance = 85
past_score = 75

new_student = np.array([[hours, attendance, past_score]])
predicted_score = model.predict(new_student)

print("\n===== NEW STUDENT PREDICTION =====")
print(f"Hours Studied: {hours}")
print(f"Attendance: {attendance}%")
print(f"Past Score: {past_score}")
print(f"Predicted Final Score: {predicted_score[0]:.2f}")