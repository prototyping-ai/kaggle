import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Define the same model architecture used during training
class CardioNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load the saved scaler
scaler = joblib.load("scaler.save")

# Load the saved model
input_dim = 10  # number of features used
model = CardioNet(input_dim=input_dim)
model.load_state_dict(torch.load("cardio_model.pt"))
model.eval()



user_input = {
    "age": 65,
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 110.5,
    "bmi": 27.8,
    "gender": 0,            # e.g., 0 = Male, 1 = Female (factorized)
    "ever_married": 1,      # 1 = Yes
    "work_type": 2,
    "Residence_type": 1,
    "smoking_status": 0
}

# Smoking Status: formerly smoker, unknown, never smoked, smokes, 
# Residence   Urban/Rural
# work_type   Private, Self-employed

# Convert to DataFrame and scale
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    prediction = model(input_tensor).item()

print(f"Stroke risk prediction (probability): {prediction:.4f}")
print("Risk Level:", "High" if prediction > 0.5 else "Low")


