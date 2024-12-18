import torch
import torch.nn as nn


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
