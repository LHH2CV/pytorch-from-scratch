import torch
import torch.nn as nn
import torch.nn.functional as F
class Multihead_Attentionlayer(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.head_k = d_model // num_heads #int
        self.head_n = num_heads
        self.d_model = d_model
        self.q_layer = nn.Linear(d_model,d_model)
        self.k_layer = nn.Linear(d_model,d_model)
        self.v_layer = nn.Linear(d_model,d_model)
        self.output_layer = nn.Linear(d_model,d_model)
    def forward(self,x):
        B,L_s,d_model = x.size()
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.k_layer(x)
        def split_heads(x):
            return x.reshape(B,L_s,self.head_n,self.head_k).transpose(1,2)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v) #b,h,n_s,dk
        attention_scores = q@k.transpose(-1,-2)/(self.head_n**0.5) #b,h,n_s,n_s
        attention_weights = F.softmax(attention_scores,-1)
        attention_outputs = attention_weights@v #b,h,n_s,dk
        attention_outputs = attention_outputs.transpose(1,2).reshape(B,L_s,d_model)
        out = self.output_layer(attention_outputs)
        return out










mha = Multihead_Attentionlayer(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # Batch=2, SeqLen=10, d_model=512
out = mha(x)
print(out.shape)  # should be [2, 10, 512]
