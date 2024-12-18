import matplotlib.pyplot as plt
import joblib
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from logger import setup_logger
import argparse
from models.attention_model import AttentionModel
from models.lstm import LSTMModel
from utils import merge_and_stat_label, filter_by_min_count, scale_df_with_scalar_multiplication, scale_df_with_MinMaxScaler

idx_range = (0, 50000)

parser = argparse.ArgumentParser(
    description="Train a model with specified epochs.")
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs for training')
parser.add_argument('--model', type=str, default='lstm',
                    help='Model to use for training',
                    choices=['lstm', 'attention'])
parser.add_argument('--dataset-path', type=str,
                    default='Datasets/TabularIoTAttacks-2024', help='Path to the dataset')
parser.add_argument('--scaler', type=str, default='minmax', help='Scaler to use for scaling the data', choices=['minmax', 'scalar'])
args = parser.parse_args()

dataset_path = args.dataset_path
# selected_attacks = ['Recon OS Scan', 'Benign Traffic']
selected_attacks = ['DoS TCP Flood', 'Recon Port Scan', 'Recon OS Scan',
                    'MQTT DDoS Publish Flood', 'MQTT DoS Connect Flood', 'Benign Traffic']

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'./logs/{timestamp}'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
logger = setup_logger(log_file=os.path.join(log_dir, 'train.log'))

logger.info(f"Log directory: {log_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


merged_df, label_stats = merge_and_stat_label(
    dataset_path, to_include=selected_attacks)
min_count = 32620
filtered_df = filter_by_min_count(merged_df, 'Attack Name', min_count)

y = filtered_df['Attack Name']
X = filtered_df.select_dtypes(include=['int64', 'float64'])
X = X.drop(columns=['Label', 'Src Port', 'Dst Port', 'Protocol'], axis=1)

if args.scaler == 'minmax':
    scalers = [MinMaxScaler(feature_range=idx_range) for _ in range(X.shape[1])]
    scaled_X = scale_df_with_MinMaxScaler(X, scalers)
elif args.scaler == 'scalar':
    scaled_X, scalars = scale_df_with_scalar_multiplication(X, *idx_range)
    logger.info(f"Scalars: {scalars}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

logger.info(f"Label Encoding: {
            dict(zip(le.classes_, le.transform(le.classes_)))}")

X_tensor = torch.tensor(scaled_X.to_numpy(), dtype=torch.long)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor,
                                                    test_size=0.4, random_state=42, stratify=y_tensor)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                test_size=0.5, random_state=42, stratify=y_temp)

hidden_size = 64
num_layers = 2
output_size = len(np.unique(y_encoded))
bidirectional = False

X_embedding_dims = 64
column_embedding_dim = 64
input_size = 128  # X_embedding_dims + column_embedding_dim

if args.model == 'lstm':
    model = LSTMModel(input_size, hidden_size, num_classes=output_size,
                      num_layers=num_layers, bidirectional=bidirectional,
                      n_features=X.shape[1], X_embedding_dims=X_embedding_dims,
                      column_embedding_dim=column_embedding_dim, X_range=idx_range)

elif args.model == 'attention':
    model = AttentionModel(
        output_size, X.shape[1], X_embedding_dims, column_embedding_dim, idx_range, num_heads=1)
model.to(device)
logger.info(f"Model information: {model}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=2, verbose=True, min_lr=1e-7)


train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

batch_size = 512 * 2
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

np.save(os.path.join(log_dir, 'y_encoded.npy'), y_encoded)

joblib.dump(le, os.path.join(log_dir, 'label_encoder.joblib'))
if args.scaler == 'minmax':
    joblib.dump(scalers, os.path.join(log_dir, 'min_max_scalers.joblib'))
elif args.scaler == 'scalar':
    np.save(os.path.join(log_dir, 'scalars.npy'), scalars)

logger.info("Training started.")
best_val_accuracy = 0.0
num_epochs = args.epochs
patience = 5
no_improve_epochs = 0

epoch_class_accuracy_train = pd.DataFrame()
epoch_class_accuracy_val = pd.DataFrame()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct_train = 0
    total_train = 0

    train_class_correct = [0] * output_size
    train_class_total = [0] * output_size

    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            for i in range(output_size):
                class_mask = targets == i
                train_class_correct[i] += (predicted[class_mask]
                                           == i).sum().item()
                train_class_total[i] += class_mask.sum().item()

            pbar.set_postfix(loss=epoch_loss / len(train_loader),
                             accuracy=100 * correct_train / total_train)

    train_accuracy = 100 * correct_train / total_train
    logger.info(f"Epoch: {epoch}/{num_epochs}, Train Loss: {epoch_loss /
                len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    train_class_accuracy = {}
    train_class_accuracy['epoch'] = epoch
    train_class_accuracy.update({le.classes_[i]: (
        train_class_correct[i] / train_class_total[i] * 100 if train_class_total[i] > 0 else 0) for i in range(output_size)})
    train_class_accuracy['overall'] = train_accuracy
    train_class_accuracy['loss'] = epoch_loss / len(train_loader)

    epoch_class_accuracy_train = pd.concat(
        [epoch_class_accuracy_train, pd.DataFrame(train_class_accuracy, index=[epoch])])
    logger.info(f"Epoch: {
                epoch}/{num_epochs}, Train Class-wise Accuracy:\n {epoch_class_accuracy_train}")

    writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0

    val_class_correct = [0] * output_size
    val_class_total = [0] * output_size

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()

            for i in range(output_size):
                class_mask = targets == i
                val_class_correct[i] += (predicted[class_mask]
                                         == i).sum().item()
                val_class_total[i] += class_mask.sum().item()

    val_accuracy = 100 * correct_val / total_val
    lr_scheduler.step(val_loss)
    logger.info(
        f"Epoch: {epoch}/{num_epochs}, learning rate: {lr_scheduler.get_last_lr()}")
    logger.info(f"Epoch: {epoch}/{num_epochs}, Validation Loss: {val_loss /
                len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    val_class_accuracy = {"epoch": epoch}
    val_class_accuracy.update({le.classes_[i]: (
        val_class_correct[i] / val_class_total[i] * 100 if val_class_total[i] > 0 else 0) for i in range(output_size)})
    val_class_accuracy['overall'] = val_accuracy
    val_class_accuracy['loss'] = val_loss / len(val_loader)

    epoch_class_accuracy_val = pd.concat(
        [epoch_class_accuracy_val, pd.DataFrame(val_class_accuracy, index=[epoch])])
    logger.info(f"Epoch: {
                epoch}/{num_epochs}, Validation Class-wise Accuracy:\n {epoch_class_accuracy_val}")

    writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improve_epochs = 0
        model_save_path = os.path.join(log_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"New best model saved at {model_save_path}")
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        logger.info(
            "Early stopping due to no improvement in validation accuracy.")
        break

train_accuracy_csv_path = os.path.join(log_dir, 'train_class_accuracy.csv')
val_accuracy_csv_path = os.path.join(log_dir, 'val_class_accuracy.csv')
epoch_class_accuracy_train.to_csv(train_accuracy_csv_path, index=False)
epoch_class_accuracy_val.to_csv(val_accuracy_csv_path, index=False)

logger.info(f"Training class accuracy saved to {train_accuracy_csv_path}")
logger.info(f"Validation class accuracy saved to {val_accuracy_csv_path}")


model.eval()
correct_test = 0
total_test = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in tqdm(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_test += targets.size(0)
        correct_test += (predicted == targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

# Save all_preds and all_labels
np.save(os.path.join(log_dir, 'all_preds.npy'), all_preds)
np.save(os.path.join(log_dir, 'all_labels.npy'), all_labels)

test_accuracy = 100 * correct_test / total_test
logger.info(f"Test Accuracy: {test_accuracy:.2f}%")

logger.info("Classification Report on Test Set:")
logger.info(classification_report(all_labels, all_preds,
            target_names=le.classes_, digits=4))

writer.add_scalar('Accuracy/test', test_accuracy)
writer.close()

ax = y.value_counts().plot(kind='bar')

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.xticks(rotation=45)

plt.savefig("bar_dist.png", bbox_inches='tight')
plt.close()

value_counts = y.value_counts()

plt.figure(figsize=(6, 6))
plt.pie(value_counts, labels=value_counts.index,
        autopct='%1.1f%%', startangle=90)

plt.title("Category Distribution")

plt.savefig("pie_dist.png", bbox_inches='tight')
plt.close()
