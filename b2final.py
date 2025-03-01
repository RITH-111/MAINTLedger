import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


# Define the SimpleModel class before loading the model
class SimpleModel:
    """
    A simple decision model that calculates failure based on weighted thresholds.
    """

    def __init__(self, thresholds, weights):
        self.thresholds = thresholds
        self.weights = weights

    def predict_proba(self, X):
        """
        Calculate probabilities based on input features.
        """
        scores = np.dot((X > self.thresholds).astype(int), self.weights)
        probabilities = 1 / (1 + np.exp(-scores))  # Sigmoid for probability
        return probabilities

    def predict(self, X, threshold=0.5):
        """
        Predict binary output based on threshold.
        """
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)


# Generate synthetic data for training
def generate_synthetic_data():
    """
    Generates a fake dataset for demonstration purposes.
    """
    np.random.seed(42)
    data = {
        "temperature": np.random.uniform(20, 80, size=1000),
        "speed": np.random.uniform(500, 1500, size=1000),
        "torque": np.random.uniform(50, 200, size=1000),
        "wear": np.random.uniform(0, 50, size=1000),
        "Machine failure": np.random.randint(0, 2, size=1000),
    }
    return pd.DataFrame(data)


# Train and save the model if it doesn't exist
def train_and_save_model():
    """
    Train and save the SimpleModel to a file.
    """
    df = generate_synthetic_data()  # Use synthetic data
    X = df.drop(columns=["Machine failure"]).values
    y = df["Machine failure"].values

    # Define thresholds based on the median value of each feature
    thresholds = np.median(X, axis=0)

    # Define weights based on correlations between features and the target
    weights = [
        np.corrcoef(X[:, i], y)[0, 1] if not np.isnan(np.corrcoef(X[:, i], y)[0, 1]) else 0
        for i in range(X.shape[1])
    ]

    # Create a SimpleModel object
    model = SimpleModel(thresholds=thresholds, weights=weights)

    # Save the model using joblib
    joblib.dump(model, "random_forest_model.pkl")
    print("Model trained and saved as random_forest_model.pkl")


# Ensure model file exists, train if needed
try:
    rf_model = joblib.load("random_forest_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model not found. Training a new model...")
    train_and_save_model()
    rf_model = joblib.load("random_forest_model.pkl")

# Define FastAPI app
app = FastAPI()


# Define the data model for requests
class MachineData(BaseModel):
    temperature: float
    speed: float
    torque: float
    wear: float


# Endpoint for predictions
@app.post("/predict/")
async def predict(machine_data: MachineData):
    """
    Predict machine failure based on input data.
    """
    # Input features in the correct order
    input_data = np.array([
        machine_data.temperature,
        machine_data.speed,
        machine_data.torque,
        machine_data.wear,
    ]).reshape(1, -1)  # Reshape for a single prediction

    probability = rf_model.predict_proba(input_data)[0]
    prediction = rf_model.predict(input_data)[0]

    return {
        "failurePredicted": bool(prediction),
        "confidence": round(probability * 100, 2),
        "maintenanceRecommended": "Yes" if prediction == 1 else "No",
    }


if __name__ == "__main__":
    import uvicorn

    # Run the app on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
