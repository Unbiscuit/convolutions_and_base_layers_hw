import torch
from torch import Tensor

class customDropout:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def forward(self, X: Tensor) -> Tensor:
        mask = (torch.rand(X.shape) > self.p).float()
        return X*mask

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)