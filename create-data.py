import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from datasets import Dataset
from datetime import datetime

# Load CSV files into a DataFrame
def merge_and_stat_label(folder_path):
    all_dfs = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)
    merged_df = pd.concat(all_dfs, ignore_index=True)
    label_stats = merged_df['Label'].value_counts()
    return merged_df, label_stats

# Filter the dataset based on minimum label count
def filter_by_min_count(df, label_column, min_count):
    label_counts = df[label_column].value_counts()
    valid_labels = label_counts[label_counts >= min_count].index
    return df[df[label_column].isin(valid_labels)]

# Create Hugging Face Dataset from raw data
def create_huggingface_dataset(X, y, X_matrix, column_matrix):
    data = []
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Process each row of data
    for i in tqdm(range(len(X))):
        x_value = X[i]  # Row in X
        attack_name = y_encoded[i]
        
        # Map X[i] values to X_matrix and column_matrix (index-based)
        X_transformed = np.concatenate([X_matrix[x_value], column_matrix])
        
        data.append({"features": X_transformed, "label": attack_name})

    return Dataset.from_dict({
        "features": [x["features"] for x in data],
        "label": [x["label"] for x in data]
    }), y_encoded, le

# Custom dataset for PyTorch DataLoader
class CustomDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features = self.dataset[idx]["features"]
        label = self.dataset[idx]["label"]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Example usage with your directory and data
folder_path = 'Datasets/TabularIoTAttacks-2024'

# Load and filter dataset
merged_df, label_stats = merge_and_stat_label(folder_path)
min_count = 32620
filtered_df = filter_by_min_count(merged_df, 'Attack Name', min_count)

# Extract X and y
y = filtered_df['Attack Name']
X = filtered_df.select_dtypes(include=['int64', 'float64'])

# Preprocess X using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 50000))
scaled_X = scaler.fit_transform(X)

# Create random X_matrix and column_matrix (example, replace with actual data)
X_matrix = np.random.rand(50000, 64)  # Replace with actual X_matrix
column_matrix = np.random.rand(76, 64)  # Replace with actual column_matrix

# Create Hugging Face dataset
hf_dataset, y_encoded, le = create_huggingface_dataset(scaled_X, y, X_matrix, column_matrix)

# Create DataLoader for training
train_dataset = CustomDataset(hf_dataset)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.fc(out)
        return out

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=128, hidden_size=64, num_classes=6)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(torch.tensor(scaled_X, dtype=torch.float32), torch.tensor(y_encoded, dtype=torch.long), test_size=0.4, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Create DataLoaders for train, validation, and test
train_loader = DataLoader(CustomDataset(hf_dataset), batch_size=64, shuffle=True)
val_loader = DataLoader(CustomDataset(hf_dataset), batch_size=64, shuffle=False)
test_loader = DataLoader(CustomDataset(hf_dataset), batch_size=64, shuffle=False)

# TensorBoard setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'./logs/{timestamp}'
os.makedirs(log_dir, exist_ok=True)

# TensorBoard writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct_train = 0
    total_train = 0
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            pbar.set_postfix(loss=epoch_loss / (total_train / 64), accuracy=100 * correct_train / total_train)
            pbar.update(1)
    
    train_accuracy = 100 * correct_train / total_train
    print(f"Train Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
    writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    # Validation step
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
    
    val_accuracy = 100 * correct_val / total_val
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    # Save model checkpoint
    model_save_path = os.path.join(log_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

# Test the model after training
model.eval()
correct_test = 0
total_test = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_test += targets.size(0)
        correct_test += (predicted == targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("Classification Report on Test Set:")
print(classification_report(all_labels, all_preds))

# Log test accuracy to TensorBoard
writer.add_scalar('Accuracy/test', test_accuracy)
writer.close()

# Save matrices to log_dir
np.save(os.path.join(log_dir, 'X_matrix.npy'), X_matrix)
np.save(os.path.join(log_dir, 'column_matrix.npy'), column_matrix)
