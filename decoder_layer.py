import torch
import torch.nn as nn
import torch.nn.functional as F

class transformer_decoder_layer(nn.Module):
    def __init__(self, d_module,d_ff,dropout,heads):
        super().__init__()
        self.atten1 = nn.MultiheadAttention(d_module,heads,dropout)
        self.atten2 = nn.MultiheadAttention(d_module,heads,dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_module,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_module)
        )
        self.layernorm1 = nn.LayerNorm(d_module)
        self.layernorm2 = nn.LayerNorm(d_module)
        self.layernorm3 = nn.LayerNorm(d_module)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,memory):
        attn_ouput,_ = self.atten1(x,x,x)
        x = self.layernorm1(x + self.dropout(attn_ouput))

        attn_ouput2,_ = self.atten2(x,memory,memory)
        x = self.layernorm2(x + self.dropout(attn_ouput2))  

        ffn_output = self.ffn(x)
        x =   self.layernorm3(x + self.dropout(ffn_output))  
        return x