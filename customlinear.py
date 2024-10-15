import torch
from torch import Tensor

class customLinear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.w = torch.randn((in_features, out_features))
        self.bias = bias

        if bias:
            self.b = torch.zeros(out_features)
        else:
            self.b = None

    def forward(self, X: Tensor) -> Tensor:
        if self.bias:
            return X@self.w + self.b
        else:
            return X@self.w
    
    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)


    