import torch
from torch import Tensor

class customBatchNorm:
    def __init__(self, num_features: int, eps:float = 1e-5, momentum:float = 0.1, affine:bool = True) -> None:
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.num_features = num_features

        if self.affine:
            self.gamma = torch.ones(num_features)  
            self.beta = torch.zeros(num_features) 
        else:
            self.gamma = None
            self.beta = None

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, X: Tensor, training:bool) -> Tensor:
        if training:
            batch_mean = torch.mean(X, dim=0)
            batch_var = torch.var(X, dim=0, unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        inputs_normalized = (X - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            inputs_normalized = self.gamma * inputs_normalized + self.beta

        return inputs_normalized

    def __call__(self, X: Tensor, training:bool = True) -> Tensor:
        return self.forward(X, training=training)
    

