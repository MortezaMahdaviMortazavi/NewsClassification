import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from feature_extraction import Embedding, EmbeddingWithPositionalEncoding
from dataclasses import dataclass

@dataclass
class AttentionConfig:
    pass


class RNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            embed_proj_size,
            rnn_hidden_size,
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
    

class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size,block_size,dropout=0.2):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, _input):
        B,T,C = _input.shape
        k = self.key(_input)
        q = self.query(_input)
        wei = q @ k.transpose(-2, -1)

        # tril = torch.tril(torch.ones(_input.size(1), _input.size(1)))
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(_input)
        out = wei @ v
        return out
    
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size,embed_size,block_size) -> None:
        super().__init__()
        """multiple attention heads in parallel"""

        """Args:
                num_heads:number of Attention heads that used in MultiHeadAttention
                head_size:output dim of key and query
                embed_size:Embedding dimension
        """
        self.heads = nn.ModuleList([
            AttentionHead(embed_size=embed_size,head_size=head_size,block_size=block_size)
            for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_size,embed_size) # projection layer followed by MHA

    def forward(self,X):
        out = torch.cat([h(X) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embed,dropout=0.2):
        super().__init__()
        """Simple Position-wise FeedForward Neural Network for our transformer block"""
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed*4),
            nn.GELU(),
            nn.Linear(n_embed*4,n_embed),
            nn.Dropout(dropout)
        )

    def forward(self,X):
        out = self.net(X)
        return out
    
class AddNorm(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_embed)

    def forward(self, x, sub_layer_output):
        # Add and normalize
        # LayerNorm(x + Sublayer(x))
        output = self.layer_norm(x + sub_layer_output)
        return output
    

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

