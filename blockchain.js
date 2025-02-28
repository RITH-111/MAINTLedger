import Web3 from 'web3';
import fs from 'fs';
import axios from 'axios';

// Connect to Ganache
const web3 = new Web3('http://127.0.0.1:7545');

// Load deployment details
const deploymentData = JSON.parse(fs.readFileSync('deployment.json', 'utf8'));
const contractAddress = deploymentData.address;
const contractABI = deploymentData.abi;

// Create contract instance
const contract = new web3.eth.Contract(contractABI, contractAddress);

// ML Model endpoint (replace with your actual ML model endpoint)
const ML_ENDPOINT = 'http://localhost:5000/predict';

async function getMachineData(index) {
    try {
        const machineData = await contract.methods.getMachine(index).call();
        return {
            id: parseInt(machineData.id),
            productID: machineData.productID,
            mType: machineData.mType,
            temperature: parseInt(machineData.temperature),
            speed: parseInt(machineData.speed),
            torque: parseInt(machineData.torque),
            wear: parseInt(machineData.wear),
            failure: machineData.failure,
            ipfsHash: machineData.ipfsHash
        };
    } catch (error) {
        console.error(`Error fetching machine data at index ${index}:`, error);
        return null;
    }
}

async function predictMaintenance(machineData) {
    try {
        // Prepare data for ML model
        const mlInput = {
            temperature: machineData.temperature,
            speed: machineData.speed,
            torque: machineData.torque,
            wear: machineData.wear
        };

        // Make prediction request to ML model
        const response = await axios.post(ML_ENDPOINT, mlInput);
        return response.data;
    } catch (error) {
        console.error('Error making ML prediction:', error);
        return null;
    }
}

async function storeToIPFS(data) {
    try {
        // Replace with your IPFS node endpoint
        const ipfs = await import('ipfs-http-client');
        const client = ipfs.create('http://localhost:5001');
        
        // Store data to IPFS
        const result = await client.add(JSON.stringify(data));
        return result.path;
    } catch (error) {
        console.error('Error storing to IPFS:', error);
        return null;
    }
}

async function updateMachineStatus(machineData, prediction, ipfsHash) {
    try {
        const accounts = await web3.eth.getAccounts();
        
        // Update machine data with new prediction
        await contract.methods.addMachine(
            machineData.id,
            machineData.productID,
            machineData.mType,
            machineData.temperature,
            machineData.speed,
            machineData.torque,
            machineData.wear,
            prediction.failurePredicted,  // Update failure status based on prediction
            ipfsHash                      // New IPFS hash with prediction data
        ).send({
            from: accounts[0],
            gas: 300000
        });

        console.log(`Updated machine ${machineData.id} with new prediction data`);
    } catch (error) {
        console.error('Error updating machine status:', error);
    }
}

async function processMachine(index) {
    try {
        // 1. Get machine data from blockchain
        console.log(`Processing machine at index ${index}...`);
        const machineData = await getMachineData(index);
        if (!machineData) return;

        // 2. Make maintenance prediction
        console.log('Making maintenance prediction...');
        const prediction = await predictMaintenance(machineData);
        if (!prediction) return;

        // 3. Store prediction results to IPFS
        const predictionData = {
            machineData: machineData,
            prediction: prediction,
            timestamp: new Date().toISOString()
        };
        
        console.log('Storing prediction data to IPFS...');
        const ipfsHash = await storeToIPFS(predictionData);
        if (!ipfsHash) return;

        // 4. Update blockchain with new status
        console.log('Updating blockchain with new status...');
        await updateMachineStatus(machineData, prediction, ipfsHash);

        console.log('\nPrediction Results:');
        console.log('-------------------');
        console.log(`Machine ID: ${machineData.id}`);
        console.log(`Failure Predicted: ${prediction.failurePredicted}`);
        console.log(`Confidence: ${prediction.confidence}%`);
        console.log(`Maintenance Recommended: ${prediction.maintenanceRecommended}`);
        console.log(`New IPFS Hash: ${ipfsHash}`);
        console.log('-------------------\n');

    } catch (error) {
        console.error(`Error processing machine ${index}:`, error);
    }
}

async function monitorMachines() {
    try {
        console.log('Starting machine monitoring...');
        
        // Process first 10 machines (adjust range as needed)
        for (let i = 0; i < 10; i++) {
            await processMachine(i);
        }
        
        console.log('Completed machine monitoring cycle');
    } catch (error) {
        console.error('Error in monitoring cycle:', error);
    }
}

// Add monitoring interval (e.g., every 5 minutes)
const MONITORING_INTERVAL = 5 * 60 * 1000; // 5 minutes in milliseconds

console.log('Starting ML Integration Service...');
monitorMachines();  // Initial run
setInterval(monitorMachines, MONITORING_INTERVAL);  // Periodic monitoring
