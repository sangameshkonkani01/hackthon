# save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load your data
data = pd.read_csv("data/diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Create models folder if it doesn't exist
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model and scaler
with open(os.path.join(MODEL_DIR, "diabetes_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "diabetes_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
