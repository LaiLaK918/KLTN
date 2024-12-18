import torch
import torch.nn as nn
from .embedding import CustomEmbedding

class AttentionModel(nn.Module):
    def __init__(self, num_classes, 
                 n_features, X_embedding_dims, column_embedding_dim,
                 X_range: tuple, num_heads=4, attention_dim=None, dropout=0.2):
        super(AttentionModel, self).__init__()
        _, X_max = X_range
        X_matrix = torch.rand(X_max + 1, X_embedding_dims)
        column_matrix = torch.rand(n_features, column_embedding_dim)
        
        self.embedding = CustomEmbedding(X_matrix, column_matrix)
        
        # If attention_dim is not provided, use concatenated embedding dimension
        self.attention_dim = attention_dim or (X_embedding_dims + column_embedding_dim)
        
        # Input projection is only necessary if attention_dim != embedding_dim
        if attention_dim and attention_dim != (X_embedding_dims + column_embedding_dim):
            self.input_proj = nn.Linear(X_embedding_dims + column_embedding_dim, attention_dim)
        else:
            self.input_proj = None
        
        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.attention_dim, 
                                                    num_heads=num_heads, 
                                                    dropout=dropout,
                                                    batch_first=True)
        
        self.fc = nn.Linear(self.attention_dim, num_classes)
        self.dropout = nn.Dropout(dropout) # Dropout layer for overfitting prevention
        
    def forward(self, indices):
        # Process inputs through the embedding layer
        x = self.embedding(indices)  # (batch_size, seq_len, embedding_dim)
        
        # Optional projection if attention_dim is different
        if self.input_proj:
            x = self.input_proj(x)  # (batch_size, seq_len, attention_dim)
        
        # Multihead Attention
        attn_output, _ = self.multihead_attn(x, x, x)  # Self-attention: Q=K=V=x
        
        # Mean pooling over sequence length
        out = attn_output.mean(dim=1)  # (batch_size, attention_dim)
        
        # Fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        return out
