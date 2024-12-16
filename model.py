import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomEmbedding(nn.Module):
    def __init__(self, X_matrix: torch.Tensor, column_matrix: torch.Tensor):
        super(CustomEmbedding, self).__init__()
        self.X_matrix = nn.Parameter(X_matrix, requires_grad=False)
        self.column_matrix = nn.Parameter(column_matrix, requires_grad=False)

    def forward(self, indices: torch.Tensor):
        """
        Apply the embedding process for input indices using batch processing.
        """
        batch_size, seq_len = indices.shape
        feature_dim_X = self.X_matrix.shape[1]
        feature_dim_C = self.column_matrix.shape[1]

        # Efficient batch operation: get X_matrix and column_matrix embeddings
        X_embeddings = self.X_matrix[indices]  # (batch_size, seq_len, feature_dim_X)
        
        # Expand column_matrix to match batch size and seq_len
        column_embeddings = self.column_matrix.unsqueeze(0).expand(batch_size, seq_len, feature_dim_C)
        
        # Concatenate along the last dimension (features dimension)
        result = torch.cat((X_embeddings, column_embeddings), dim=-1)

        return result

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, 
                 n_features, X_embedding_dims, column_embedding_dim,
                 X_range: tuple,
                 num_layers=2, dropout=0.2, **kwargs):
        super(LSTMModel, self).__init__()
        X_min, X_max = X_range
        X_matrix = torch.rand(X_max + 1, X_embedding_dims)
        column_matrix = torch.rand(n_features, column_embedding_dim)
        self.embedding = CustomEmbedding(X_matrix, column_matrix)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if kwargs.get('bidirectional') else 1
        
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, **kwargs)

        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, indices):
        # Process inputs through the embedding layer
        x = self.embedding(indices)

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        # LSTM output
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last time-step output

        # Pass through fully connected layer
        out = self.fc(out)
        return out

