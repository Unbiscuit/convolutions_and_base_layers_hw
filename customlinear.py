import torch
from torch import Tensor

class customLinear():
    def __init__(self, in_features: int, out_features: int) -> None:
        self.w = torch.rand((in_features, out_features))
        self.b = torch.rand(out_features)

    def forward(self, X: Tensor) -> Tensor:

        return X@self.w + self.b
    
    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)


    