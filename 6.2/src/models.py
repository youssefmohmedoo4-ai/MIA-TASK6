import torch
import torch.nn as nn
from src.config import Config

class EncoderCNN(nn.Module):
    """Projects extracted 2048-d CNN features to match the LSTM Embedding space."""
    def __init__(self, feature_dim=Config.FEATURE_DIM, embed_dim=Config.EMBED_DIM, dropout=Config.DROPOUT):
        super().__init__()
        self.fc = nn.Linear(feature_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, features):
        x = self.fc(features)
        x = self.bn(x)
        x = self.relu(x)
        return self.dropout(x)

class DecoderLSTM(nn.Module):
    """Autoregressive LSTM decoder for caption sequence generation."""
    def __init__(
        self, 
        vocab_size, 
        embed_dim=Config.EMBED_DIM, 
        hidden_dim=Config.HIDDEN_DIM, 
        num_layers=Config.NUM_LAYERS, 
        dropout=Config.DROPOUT,
        pad_idx=0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, projected_features, captions):
        # embeddings: (B, Seq_Len - 1, Embed_Dim)
        embeddings = self.dropout(self.embedding(captions[:, :-1]))
        
        # Concatenate projected image feature as the initial sequence token
        # inputs: (B, Seq_Len, Embed_Dim)
        inputs = torch.cat((projected_features.unsqueeze(1), embeddings), dim=1)
        
        hiddens, _ = self.lstm(inputs)
        outputs = self.fc(self.dropout(hiddens))
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, pad_idx=0):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderLSTM(vocab_size=vocab_size, pad_idx=pad_idx)

    def forward(self, features, captions):
        projected = self.encoder(features)
        outputs = self.decoder(projected, captions)
        return outputs