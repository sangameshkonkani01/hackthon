import pickle
import numpy as np

# Load the trained model and scaler
with open("models/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/diabetes_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Feature order
features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

# Take input from user
inputs = []
for feature in features:
    val = float(input(f"Enter {feature}: "))
    inputs.append(val)

# Prepare input for prediction
X = np.array(inputs).reshape(1, -1)
X = scaler.transform(X)  # Scale input

# Predict
probability = model.predict_proba(X)[0][1]
prediction = model.predict(X)[0]

# Convert probability to risk
if probability < 0.4:
    risk = "Low"
elif probability < 0.7:
    risk = "Medium"
else:
    risk = "High"

# Output
print(f"\nPredicted risk: {risk}")
print(f"Probability of having diabetes: {probability:.2f}")
