#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from enum import Enum
from multiprocessing.sharedctypes import Value
from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor

import torch.nn.functional as F


from . import transform_signal as ts

def draw_random_float(min, max, size = (1,)) -> Tensor:
    return torch.rand(size) * (max - min) + min

def _apply_op(
    x: Tensor, op_name: str, magnitude: float,  fill: Optional[List[float]]
):
    if op_name == "Jitter":
        x = ts.jitter(x, magnitude)
    elif op_name == "Scaling":
        x = ts.scaling(x,magnitude)
    elif op_name == "Flip_Y":
        x = ts.flip_y(x)
    elif op_name == "Window_slice":
        x = ts.window_slice(x)
    elif op_name == "Drop_block":
        x = ts.drop_block(x, magnitude)
    elif op_name == "Random_block":
        x = ts.random_block(x, magnitude)
    elif op_name == "Shuffle_block":
        x = ts.random_block_shuffle(x, magnitude)
        
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return x

class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 1,
        magnitude: int = 3,
        augmentation_operations = None,
        num_magnitude_bins: int = 10+1,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.augmentation_operations = augmentation_operations
        self.fill = fill

        
        if (self.magnitude < 1) | (self.magnitude > 10):
            raise ValueError(f"Magnitude must be less than 10. Got {self.magnitude}.")

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        dict_op = {
            # op_name: (magnitudes, signed (can be pos or neg))
            "Identity"      : (torch.tensor(0.0), False),
            "Jitter"        : (torch.linspace(0.0, 0.8, num_bins), False),
            "Scaling"       : (torch.linspace(0.0, 0.5, num_bins), False),
            "Flip_Y"        : (torch.tensor(0.0), False),
            "Window_slice"  : (1-torch.linspace(0.0, 0.6, num_bins), False),
            "Drop_block"    : (torch.linspace(0.0, 0.8, num_bins), False),
            "Random_block"  : (torch.linspace(0.0, 0.8, num_bins), False),
            "Permute_block" : (torch.linspace(0.0, 0.8, num_bins), False),
            "Shuffle_block" : (torch.linspace(0.0, 0.8, num_bins), False),
         
        }
        if self.augmentation_operations is not None:
            dict_op = {key: value for key, value in dict_op.items() if key in self.augmentation_operations}
        return dict_op

    def forward(self, x: Tensor, nb_epoch) -> Tensor:
        """
            x : Signal to be transformed.

        Returns:
            x : Tensor: Transformed signal.
        """
        fill = self.fill
        # channels, height, width = F.get_dimensions(img)
        # if isinstance(img, Tensor):
            # if isinstance(fill, (int, float)):
                # fill = [float(fill)] * channels
            # elif fill is not None:
                # fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]

            if nb_epoch <15:
                bounded_magnitude = int((nb_epoch/25) * self.magnitude)
            else:
                bounded_magnitude = self.magnitude

            magnitude = float(draw_random_float(magnitudes[0], magnitudes[bounded_magnitude]).item()) if magnitudes.ndim > 0 else 0.0
            
            # print("Magnitude", magnitude, "op_name", op_name)
            # if signed and torch.randint(2, (1,)):
                # magnitude *= -1.0
            x = _apply_op(x, op_name, magnitude, fill=fill)

        return x


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", fill={self.fill}"
            f")"
        )
        return s