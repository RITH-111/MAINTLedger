from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from web3 import Web3
import json
import requests

# Initialize FastAPI app
app = FastAPI()

# Load the trained ML model
rf_model = joblib.load("random_forest_model.pkl")

# Blockchain Configuration (Ganache or Infura)
INFURA_URL = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
w3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Load contract ABI and address from deployment file
with open("deployment.json", "r") as f:
    deployment_data = json.load(f)
contract_address = deployment_data["address"]
contract_abi = deployment_data["abi"]
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# IPFS Configuration (Replace with your own IPFS node if needed)
IPFS_API_URL = "http://localhost:5001/api/v0/add"

# Define expected input data format
class MachineData(BaseModel):
    temperature: float
    speed: float
    torque: float
    wear: float

# Load feature column names from dataset
dataset_path = "ai4i2020.csv"
df = pd.read_csv(dataset_path)
df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])
X_columns = df.drop(columns=["Machine failure"]).columns

# ML Prediction Function
def predict_failure(data):
    input_df = pd.DataFrame([data], columns=X_columns)
    probability = rf_model.predict_proba(input_df)[:, 1][0]
    prediction = 1 if probability > 0.3 else 0  # Threshold of 0.3

    return {
        "failurePredicted": bool(prediction),
        "confidence": round(probability * 100, 2),
        "maintenanceRecommended": "Yes" if prediction == 1 else "No"
    }

# Function to store prediction data in IPFS
def store_to_ipfs(data):
    try:
        response = requests.post(IPFS_API_URL, files={"file": json.dumps(data)})
        return response.json()["Hash"]
    except Exception as e:
        print("Error storing to IPFS:", e)
        return None

# Function to update blockchain with prediction
def update_blockchain(machine_id, prediction, ipfs_hash):
    accounts = w3.eth.accounts
    try:
        tx = contract.functions.addMachine(
            machine_id,  # Machine ID
            "PRODUCT_XYZ",  # Example Product ID
            "Type_A",  # Example Machine Type
            prediction["failurePredicted"],
            ipfs_hash
        ).build_transaction({
            'from': accounts[0],
            'gas': 300000
        })

        signed_tx = w3.eth.account.sign_transaction(tx, private_key="YOUR_PRIVATE_KEY")
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return w3.to_hex(tx_hash)
    except Exception as e:
        print("Error updating blockchain:", e)
        return None

# API Endpoint to Predict Machine Failure & Update Blockchain
@app.post("/predict/")
async def predict(machine_data: MachineData):
    # Convert input to NumPy array
    input_data = [machine_data.temperature, machine_data.speed, machine_data.torque, machine_data.wear]
    
    # Get prediction from ML model
    prediction = predict_failure(input_data)
    
    # Store results in IPFS
    ipfs_hash = store_to_ipfs(prediction)
    if not ipfs_hash:
        raise HTTPException(status_code=500, detail="Failed to store data on IPFS")

    # Update Blockchain
    tx_hash = update_blockchain(machine_id=1, prediction=prediction, ipfs_hash=ipfs_hash)
    if not tx_hash:
        raise HTTPException(status_code=500, detail="Failed to update blockchain")

    return {
        "prediction": prediction,
        "ipfs_hash": ipfs_hash,
        "tx_hash": tx_hash
    }

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
