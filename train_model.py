# train_model_best_accuracy.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv("StudentPerformanceFactors.csv")

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
# Create Study Efficiency: Hours_Studied per Sleep_Hours
data["Study_Efficiency"] = data["Hours_Studied"] / (data["Sleep_Hours"] + 0.1)  # avoid division by zero

# Encode Motivation_Level numerically
motivation_map = {"Low": 0, "Medium": 1, "High": 2}
data["Motivation_Level_Score"] = data["Motivation_Level"].map(motivation_map)

# -----------------------------
# Step 3: Numeric + Categorical Features
# -----------------------------
numeric_features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
                    "Study_Efficiency", "Motivation_Level_Score"]

categorical_features = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender"
]

# Fill missing values
data.fillna(data.mean(numeric_only=True), inplace=True)
data.fillna("Unknown", inplace=True)

# Label encode categorical features
for col in categorical_features:
    data[col] = data[col].astype(str)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Combine features
X = data[numeric_features + categorical_features]
y = data["Exam_Score"]

# -----------------------------
# Step 4: Scale numeric features
# -----------------------------
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# -----------------------------
# Step 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 6: Define Base Models
# -----------------------------
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

# -----------------------------
# Step 7: Stacking Ensemble
# -----------------------------
stack_model = StackingRegressor(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    final_estimator=XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        objective='reg:squarederror'
    ),
    passthrough=True,
    n_jobs=-1
)

# Train the stacked model
stack_model.fit(X_train, y_train)

# -----------------------------
# Step 8: Evaluate
# -----------------------------
y_pred = stack_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R2 Score: {r2:.3f}")  # Expect ~0.75-0.80

# -----------------------------
# Step 9: Save Model and Scaler
# -----------------------------
joblib.dump(stack_model, "student_score_model_best.pkl")
joblib.dump(scaler, "student_score_scaler_best.pkl")
print("Best-accuracy stacked model and scaler saved!")

# -----------------------------
# Step 10: Optional Visualization
# -----------------------------
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Student Scores")
plt.show()