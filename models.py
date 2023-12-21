import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from feature_extraction import Embedding

class RNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            embed_proj_size,
            rnn_hidden_size,
            output_size,
            is_bidirectional,
            n_layer,
            num_classes,
            dropout_rate=0.2
        ):
        super().__init__()
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embed_size=embed_size,
            proj_size=embed_proj_size,
            dropout_rate=0.2,
            use_batch_norm=False
        )
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=rnn_hidden_size,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=is_bidirectional,
            dropout=dropout_rate
        )
        self.classifier = nn.Linear(
            in_features=rnn_hidden_size * 2 if is_bidirectional else rnn_hidden_size,
            out_features=num_classes 
        )

    def forward(self, input_tensor):
        """
        Forward pass of the RNN model.

        Args:
        - input_tensor: Input tensor representing sequences of token indices.

        Returns:
        - logits: Output logits from the classifier.
        """
        embedded = self.embedding(input_tensor)

        lstm_out, _ = self.lstm(embedded)
        logits = self.classifier(lstm_out[:,-1,:])
        return logits