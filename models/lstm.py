import torch
import torch.nn as nn
from .embedding import CustomEmbedding

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

