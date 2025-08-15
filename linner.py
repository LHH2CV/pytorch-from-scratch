import torch
class Linearlayer:
    def __init__(self,input_dim,output_dim):
        self.weights = torch.randn(output_dim,input_dim)*10
        self.bias = torch.zeros(output_dim)
    def forward(self,x):
        return torch.matmul(x,self.weights.T) + self.bias
    
lin = Linearlayer(3,12)
x = torch.randn(2,3)
output = lin.forward(x)
print("输入张量 (x):")
print(x)
print("\n输出张量 (y = xW^T + b):")
print(output)
        