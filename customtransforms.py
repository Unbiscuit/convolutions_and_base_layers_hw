import torch 
import PIL
import numpy as np
from torch import Tensor
from PIL import Image

class customToTensor:
    def __call__(self, pic: Image) -> None:
        image = np.array(pic, dtype=np.float32)
        image /= 255.0
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2, 0, 1)
        
        return tensor