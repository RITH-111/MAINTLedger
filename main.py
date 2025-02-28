import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
rf_model = joblib.load("random_forest_model.pkl")

# Load dataset to get column names
dataset_name = r"D:\\MAINTLedger\\Models\\Scripts\\ai4i2020.csv"
df = pd.read_csv(dataset_name)
df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])
X_columns = df.drop(columns=["Machine failure"]).columns

def predict_machine_failure(*data, threshold=0.3):
    """Function to predict machine failure."""
    expected_columns = len(X_columns)

    if not data:
        data = np.zeros(expected_columns)  # Default to zeros if no input provided
    elif len(data) == 1 and isinstance(data[0], (list, tuple, np.ndarray)):
        data = np.array(data[0])  # Convert to NumPy array if a list or tuple is passed
    else:
        data = np.array(data)
    
    # Ensure data has the correct length
    data = np.pad(data, (0, max(0, expected_columns - len(data))), 'constant')[:expected_columns]
    
    input_df = pd.DataFrame([data], columns=X_columns)
    probability = rf_model.predict_proba(input_df)[:, 1][0]
    prediction = 1 if probability > threshold else 0

    # Probability Distribution Graph
    plt.figure(figsize=(6, 4))
    labels = ['No Failure', 'Failure']
    probabilities = [1 - probability, probability]
    plt.bar(labels, probabilities, color=['green', 'red'])
    plt.xlabel('Prediction Result')
    plt.ylabel('Probability')
    plt.title('Prediction Probability Distribution')
    plt.ylim(0, 1)
    plt.show()

    return {
        "Result": "Failure" if prediction == 1 else "No Failure",
        "Probability": probability,
        "Maintenance Required": "Maintenance Required" if prediction == 1 else "No Maintenance Required"
    }
