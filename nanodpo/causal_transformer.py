# filename: causal_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanodpo.sinusoidal_positional_encoding import SinusoidalPositionalEncoding

class CausalTransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=1024, dropout=0.1):
        super(CausalTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        q = k = src2
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class CausalTransformer(nn.Module):
    def __init__(self, d_feature, d_model, n_head, n_layer, num_actions=3, dim_feedforward=2048, dropout=0.1, 
                 max_len=5000, device=torch.device("cpu")):
        super(CausalTransformer, self).__init__()
        self.device = device  # Store the device
        self.embedding = nn.Linear(d_feature, d_model)
        self.layers = nn.ModuleList([CausalTransformerLayer(d_model, n_head, dim_feedforward, dropout) 
                                     for _ in range(n_layer)])
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len, device=self.device).to(self.device)
        self.output_layer = nn.Linear(d_model, num_actions)  # Output layer for three actions

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)

        output = self.output_layer(src)
        # Optional: Apply softmax for probabilities
        # output = F.softmax(output, dim=-1)
        output = output[:, -1, :]  # Shape: [batch_size, num_actions]
        return output
