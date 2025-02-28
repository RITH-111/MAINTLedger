import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN

# Load dataset
dataset_name = r"D:\\MAINTLedger\\Models\\Scripts\\ai4i2020.csv"
df = pd.read_csv(dataset_name)

# Drop unnecessary columns
df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

# Encode categorical column 'Type'
label_encoder = LabelEncoder()
df["Type"] = label_encoder.fit_transform(df["Type"])

# Define features (X) and target variable (y)
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using ADASYN
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

class_weights = {0: 1, 1: 10}  # Higher weight for failure cases
grid_search = GridSearchCV(RandomForestClassifier(class_weight=class_weights, random_state=42),
                           param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model after tuning
rf_model = grid_search.best_estimator_

# Save and load the trained model
joblib.dump(rf_model, "random_forest_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
