import torch as ch
import numpy as np

import os
import torch.nn as nn
from torchvision.datasets import ImageNet
from torchvision import transforms

from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from dataclasses import replace

__all__ = ['InvertTensorImageColors', 'InvertImageColorsFFCV']

class InvertTensorImageColors(nn.Module):
    def __init__(self):
        super(InvertTensorImageColors, self).__init__()

    def __call__(self, img):
        # Check if the image is in uint8 format
        if img.dtype == torch.uint8 or img.dtype == np.uint8:
            return 255 - img
        # assume the image is in float format
        else:
            return 1 - img
        
class InvertImageColorsFFCV(Operation):
    def __init__(self):
        super().__init__()

    def generate_code(self) -> Callable:
        def invert_colors(inp, dst):
            # Check if the image is in uint8 format
            if inp.dtype == ch.uint8 or inp.dtype == np.uint8:
                return 255 - inp
            # Assume the image is in float format
            else:
                return 1 - inp

        invert_colors.is_parallel = True
        return invert_colors

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # No change in dtype or jit_mode required for this operation
        return previous_state, None