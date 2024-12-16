import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

class TabularIoTAttacks2024(Dataset):
    def __init__(self, file_names: list[str], window_size: int, label_encoder=None, scaler=None):
        """
        Args:
            file_names (list of str): List of paths to CSV files.
            window_size (int): Number of elements to return in a single __getitem__ call.
            label_encoder (LabelEncoder, optional): Pre-existing LabelEncoder to encode labels.
            scaler (MinMaxScaler, optional): Pre-existing MinMaxScaler to scale features.
        """
        self.window_size = window_size
        
        original_data = pd.concat([pd.read_csv(file) for file in tqdm(file_names)], ignore_index=True)

        # Merge all CSV files into one DataFrame
        self.data = original_data.copy()

        # Remove the 'Attack Name' column and other non-numeric columns
        self.data = self.data.drop(columns=['Attack Name', 'Label', 'Src Port', 'Dst Port', 'Protocol'], errors='ignore')
        self.data = self.data.select_dtypes(exclude=['object'])

        # Use pre-existing scaler if provided, otherwise create a new one
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.data_scaled = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data_scaled = self.scaler.transform(self.data)

        # Convert to a PyTorch tensor
        self.data_tensor = torch.tensor(self.data_scaled, dtype=torch.float32)

        # If no label encoder is provided, create a new one
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(original_data['Attack Name'])
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(original_data['Attack Name'])

        # Calculate the number of windows
        self.num_windows = len(self.data) // self.window_size

    def __len__(self):
        """Returns the total number of windows in the dataset."""
        return self.num_windows

    def __getitem__(self, idx):
        """
        Returns a slice of the data tensor of size `window_size` from a valid index.
        Also returns the corresponding label.
        
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (data_tensor, label) where:
                - data_tensor is the slice of the data tensor with shape (window_size, num_features)
                - label is the label for this window (int)
        """
        start_idx = self.window_size * idx
        end_idx = start_idx + self.window_size
        
        # Get the feature data and label for the current window
        window_data = self.data_tensor[start_idx:end_idx]
        window_label = self.labels[start_idx + self.window_size - 1]  # Label of the last entry in the window
        
        return window_data, window_label

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def create_datasets(file_names, window_size, test_size=0.2, val_size=0.1, label_encoder=None, scaler=None, batch_size=64):
    """
    Creates train, validation, and test datasets from raw CSV files and splits them.
    Args:
        file_names (list): List of CSV file paths.
        window_size (int): The size of the window (sequence length).
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the dataset to include in the validation split.
        label_encoder (LabelEncoder): Pre-trained label encoder or None to create a new one.
        scaler (MinMaxScaler): Pre-trained scaler or None to create a new one.
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for the respective splits.
        scaler, label_encoder: Fitted scaler and label encoder used in data preprocessing.
    """
    # Create the full dataset object
    dataset = TabularIoTAttacks2024(file_names, window_size, label_encoder=label_encoder, scaler=scaler)

    # Split indices for train, validation, and test sets
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    # Train-test split
    train_indices, test_indices = train_test_split(indices, test_size=test_size, shuffle=True)
    
    # Train-validation split
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, shuffle=True)

    # Create the DataLoader for each split using the indices
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)

    # Return the DataLoader instances and the scaler/encoder used
    return train_loader, val_loader, test_loader, dataset.scaler, dataset.label_encoder, dataset
