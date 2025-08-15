import torch
import torch.nn as nn
import torch.nn.functional as F
class attn(nn.Module):
    def __init__(self,d_module,num_heads):
        super().__init__(attn,self)
        assert d_module % num_heads == 0
        self.d_module = d_module 
        self.numheads = num_heads
        self.d_k = d_module // num_heads
        self.q_layer = nn.Linear(d_module,d_module)
        self.k_layer = nn.Linear(d_module,d_module)
        self.v_layer = nn.Linear(d_module,d_module)
        self.ratio = d_module**0.5
        #self.atte_weights =
    def forward(self,x):
        B,L,d_module = x.size()

        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)
        def split_heads(x):
            return x.reshape(B,L,self.numheads,self.d_k).transpose(1,2)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        attention_score = q@k.transpose(-1,-2)/self.ratio #
       # attention_score = torch.bmm(q,k.transpose(-1,-2))/self.ratio 
        attention_weights = F.softmax(attention_score,-1)
        attention_output = attention_weights@v
        output = attention_output.transpose(1,2).reshape(B,L,self.d_module)
        return  attention_output