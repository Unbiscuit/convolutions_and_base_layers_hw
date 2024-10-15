import torch
from torch import Tensor

class customRelu:
    def __init__(self) -> None:
        ...

    def forward(self, X: Tensor) -> Tensor:
        return torch.maximum(X, torch.zeros(X.shape))

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)

