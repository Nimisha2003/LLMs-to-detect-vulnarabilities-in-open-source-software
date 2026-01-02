import torch
import torch.nn as nn

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, num_layers=2, hidden_dim=256, dropout=0.1):
        super(SimpleTransformerClassifier, self).__init__()

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = self.transformer(x)        # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)              # (batch, embed_dim)
        x = self.fc(x)                 # (batch, 1)
        x = self.sigmoid(x)            # (batch, 1)
        return x