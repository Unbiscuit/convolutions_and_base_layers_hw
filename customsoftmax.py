import torch
from torch import Tensor

class customSoftmax:
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        stable_X = X - torch.max(X, dim=self.dim, keepdim=True)[0]
        exp_inputs = torch.exp(stable_X)
        softmax_output = exp_inputs / torch.sum(exp_inputs, dim=self.dim, keepdim=True)
        return softmax_output

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)
    
