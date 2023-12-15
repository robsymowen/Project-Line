import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import requests
from io import BytesIO

def download_image(url="https://dl.fbaipublicfiles.com/dino/img.png"):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    
    return img
    
class ProcessLineDrawing(nn.Module):
    def __init__(self, mean, std, repeat=True, invert=True):
        super().__init__()  # Call to the superclass initializer

        # Handling single or three values for mean and std
        mean = self._expand_to_tensor(mean, 3)
        std = self._expand_to_tensor(std, 3)

        self.repeat = repeat
        self.invert = invert

        self.register_buffer('mean', mean.view(1, 3, 1, 1))
        self.register_buffer('std', std.view(1, 3, 1, 1))

    def _expand_to_tensor(self, value, length):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value).float()
        if value.numel() == 1:
            value = value.repeat(length)
        elif value.numel() != length:
            raise ValueError(f"Expected {length} elements for mean/std but got {value.numel()}")
        return value

    def forward(self, x):
        if self.repeat:
            x = x.repeat(1, 3, 1, 1)

        if self.invert:
            x = 1 - x

        x = (x - self.mean) / self.std

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean.flatten().cpu().numpy()}, std={self.std.flatten().cpu().numpy()}, repeat={self.repeat}, invert={self.invert})"

class HybridModel(nn.Module):
  def __init__(self, generator, transform, backbone):
    super().__init__()  # Call to the superclass initializer
    self.generator = generator
    self.transform = transform
    self.backbone = backbone
  
  def show_line_drawing(self, x, normalize=False):
    if len(x.shape)==3:
      x = x[0]

    if normalize:
      x = x - x.min()
      x = x / x.max()

    return Image.fromarray(x.mul(255).cpu().numpy().astype(np.uint8))

  def forward(self, x, return_line_drawings=False):
    line_drawing = self.generator(x)
    line_drawing_processed = self.transform(line_drawing)
    output = self.backbone(line_drawing_processed)

    if return_line_drawings:
      return output, line_drawing, line_drawing_processed

    return output
