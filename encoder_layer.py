import torch
import torch.nn as nn
import torch.nn.functional as F

class transformer_encoder_layer(nn.Module):
    def __init__(self,d_modules,num_heads,d_ff,dropout = 0.1):
        super().__init__()
        #att mlp resnet ln
        self.attention_layer = nn.MultiheadAttention(d_modules,num_heads,dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_modules,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_modules)
        )
        self.layernorm1 = nn.LayerNorm(d_modules)
        self.layernorm2 = nn.LayerNorm(d_modules)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        attention_outputs,attention_weights = self.attention_layer(x,x,x)
        x = self.layernorm1(x + self.dropout(attention_outputs))
        mlp_outputs = self.ffn(x)
        x = self.layernorm2(x + self.dropout(mlp_outputs))
        return x