import torch
import torch.nn as nn
import torch.nn.functional as F
class decoder_layer(nn.Module):
    def __init__(self, d_module,d_ff,num_heads,dropout):
        super().__init__(decoder_layer,self)
        self.attn = nn.MultiheadAttention(d_module,num_heads,dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_module,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_module)
        )
        self.ln1 = nn.LayerNorm(d_module)
        self.drop = nn.Dropout(dropout)
    def forward(self,x):
    #    x = self.ln1((self.attn(x)+x))
        attention_output,_ = self.attn(x)
        x = self.ln1(x + self.drop(attention_output))
        
        x = self.attn(x)

        
        return 