from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import numpy as np
from model import LSTMModel  # Import your LSTM model class
from logger import setup_logger  # Import the logger

# Set up logger
logger = setup_logger(log_file="prediction.log")

# Load the trained model, label encoder, and scaler
model = LSTMModel(input_size=10, hidden_size=64, num_classes=5)  # Adjust input_size and num_classes as per your model
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# FastAPI instance
app = FastAPI()

# Define the input data schema using Pydantic
class PredictionRequest(BaseModel):
    data: list  # List of features, expected to be a list of numbers (e.g., window size x features)

# Define the prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request with data: {request.data}")
        
        # Convert input data to numpy array and scale it using the saved scaler
        input_data = np.array(request.data)  # Assuming 'data' is a list of features
        input_data_scaled = scaler.transform(input_data.reshape(1, -1))  # Reshape to (1, num_features)
        
        # Convert the scaled input data to a PyTorch tensor
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

        # Make the prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        # Decode the predicted label back to its original form
        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())
        logger.info(f"Prediction made: {predicted_label[0]}")
        
        return {"prediction": predicted_label[0]}

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
