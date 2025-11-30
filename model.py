import torch
import torch.nn as nn

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, hidden_dim=2048, dropout=0.1):
        super(SimpleTransformerClassifier, self).__init__()

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # So inputs are (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len)
        """
        # Embed tokens
        x = self.embedding(x)

        # Pass through transformer layers
        x = self.transformer(x)

        # Take mean across sequence dimension
        x = x.mean(dim=1)

        # Classification
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

