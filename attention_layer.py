import torch
import torch.nn as nn
import torch.nn.functional as F
class simple_attention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** 0.5
    def forward(self,q,k,v):
        atten_socres = torch.bmm(q,k.transpose(1,2))/self.scale
        atten_weights = F.softmax(atten_socres,dim = -1)
        attem_output = torch.bmm(atten_weights,v)
        return attem_output,atten_weights
    
B,T,D = 2,4,8
Q= torch.randn(B,T,D)
K = torch.randn(B,T,D)
V= torch.randn(B,T,D)

AT = simple_attention(D)
print(AT.forward(Q,K,V))

