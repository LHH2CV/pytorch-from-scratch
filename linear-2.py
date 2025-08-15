import torch
class lin:
    def __init__(self,ins,ous):
        self.weights = torch.randn(ins,ous)*0.01
        self.bias = torch.zeros(ous)
    def forward(self,x):
        output = torch.matmul(x,self.weights) + self.bias
        return output