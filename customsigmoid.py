import torch
from torch import Tensor

class customSigmoid:
    def __init__(self) -> None:
        ...

    def forward(self, X: Tensor) -> Tensor:
        return torch.sigmoid(X)

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)
    

    

