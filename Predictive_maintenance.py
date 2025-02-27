import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load the dataset
df = pd.read_csv(r"C:\Users\91790\OneDrive\Documents\DATASET -1.csv")

df = df.drop(columns=["UDI", "Product ID"])  # Drop unnecessary columns

# Encode categorical column 'Type'
label_encoder = LabelEncoder()
df["Type"] = label_encoder.fit_transform(df["Type"])

# Define features (X) and target variable (y)
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model after tuning
rf_model = grid_search.best_estimator_

# Save the trained model
joblib.dump(rf_model, "random_forest_model.pkl")

# Load trained model for real-time prediction
rf_model = joblib.load("random_forest_model.pkl")

# Train Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)
joblib.dump(iso_forest, "isolation_forest_model.pkl")

# Load the anomaly detection model
iso_forest = joblib.load("isolation_forest_model.pkl")

def predict_data(data, is_real_time=True):
    """Function to predict machine failure and detect anomalies for both real-time and user-entered data."""
    input_df = pd.DataFrame([data], columns=X.columns)  # Ensure correct format
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[:, 1][0]
    
    # Check for anomalies (Real-Time and User-Entered Data)
    anomaly_score = iso_forest.decision_function(input_df)[0]
    is_anomaly = iso_forest.predict(input_df)[0] == -1
    
    # Plot the probability distribution
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
        "Data Type": "Real-Time" if is_real_time else "User-Entered",
        "Anomaly Detected": is_anomaly,
        "Anomaly Score": anomaly_score
    }
