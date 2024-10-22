import torch 
import PIL
import numpy as np
from abc import ABC, abstractmethod
from torch import Tensor
from PIL import Image
from typing import Any

class customBaseTransform(ABC):
    def __init__(self, p: float) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p

    @abstractmethod
    def forward(self, *inputs: Any) -> Any:
        ...

    def __call__(self, *inputs: Any) -> Any:
        return self.forward(*inputs)


class customToTensor:
    def __call__(self, pic: PIL.Image.Image) -> None:
        image = np.array(pic, dtype=np.float32)
        image /= 255.0
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2, 0, 1)
        
        return tensor