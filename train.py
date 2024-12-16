import torch
import torch.optim as optim
from sklearn.metrics import classification_report
import joblib
from model import LSTMModel
import torch.nn as nn
from tqdm import tqdm  # Import tqdm for progress bar
from logger import setup_logger  # Import the logger
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer
import os
from datetime import datetime

# Set up logger
logger = setup_logger(log_file="training.log")

def train_and_evaluate(model: LSTMModel, train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                       original_dataset, epochs=20, learning_rate=0.001, model_save_path="lstm_model.pth",
                       encoder_save_path="label_encoder.pkl", scaler_save_path="scaler.pkl", windows_size=128):
    # Generate a unique log directory based on timestamp
    log_dir = f"runs/lstm_training_windows_size_{windows_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        # Log the start of each epoch
        logger.info(f"Epoch [{epoch+1}/{epochs}] training started.")
        
        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                total_train_loss += loss.item()

                # Update progress bar with loss and accuracy
                pbar.set_postfix(loss=total_train_loss/len(train_loader), accuracy=100 * correct_train/total_train)

        # Log metrics to TensorBoard
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        logger.info(f"Epoch [{epoch+1}/{epochs}] training completed. Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()
        correct_val = 0
        total_val = 0
        logger.info(f"Epoch [{epoch+1}/{epochs}] validation started.")
        with torch.no_grad():
            with tqdm(val_loader, desc="Validating", unit="batch") as pbar_val:
                for inputs, labels in pbar_val:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    pbar_val.set_postfix(accuracy=100 * correct_val/total_val)

        val_accuracy = 100 * correct_val / total_val
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        logger.info(f"Epoch [{epoch+1}/{epochs}] validation completed. Accuracy: {val_accuracy:.2f}%")

    # Save model state_dict (weights) only in the same directory as logs
    model_save_path = os.path.join(log_dir, model_save_path)
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved at {model_save_path}")

    # Save the label encoder and scaler from the original dataset in the same directory
    encoder_save_path = os.path.join(log_dir, encoder_save_path)
    scaler_save_path = os.path.join(log_dir, scaler_save_path)
    joblib.dump(original_dataset.label_encoder, encoder_save_path)  # Save label encoder
    joblib.dump(original_dataset.scaler, scaler_save_path)  # Save scaler
    logger.info(f"Label encoder and scaler saved at {encoder_save_path} and {scaler_save_path}")

    # Test the model
    model.eval()
    correct_test = 0
    total_test = 0
    all_labels = []
    all_preds = []
    logger.info("Testing started.")
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as pbar_test:
            for inputs, labels in pbar_test:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Accumulate results for classification report
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                pbar_test.set_postfix(accuracy=100 * correct_test/total_test)

    test_accuracy = 100 * correct_test / total_test
    writer.add_scalar('Accuracy/Test', test_accuracy, epoch)

    logger.info(f"Test Accuracy: {test_accuracy:.2f}%")

    # Print the classification report
    logger.info("\nClassification Report:\n")
    report = classification_report(all_labels, all_preds)
    logger.info(report)
    print(report)

    # Close the TensorBoard writer
    writer.close()
    return model_save_path, encoder_save_path, scaler_save_path
