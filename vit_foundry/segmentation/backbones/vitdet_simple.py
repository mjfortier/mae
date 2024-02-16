from dataclasses import dataclass
from typing import List, Tuple
from torch import Tensor, nn
from vit_foundry.vit_components import ViTMAEConfig
from vit_foundry.segmentation.components.vitdet import ViTDet
from vit_foundry.segmentation.config import Mask2FormerBackboneOutput


@dataclass
class ViTDetBackboneConfig(ViTMAEConfig):
    # FPN parameters
    fpn_hidden_size: int = 256           # all feature pyramid entries will have this size
    fpn_layer_resolutions: List[float] = (0.5, 1.0, 2.0, 4.0) # Down / upsampling ratios for making the pyramid
    global_attention_layer_interval: int = 6 # Every n layers, we use global attention (see paper)
    window_size: int = 16


class ViTDetBackbone(nn.Module):
    def __init__(self, config: ViTDetBackboneConfig):
        super().__init__()
        self.config = config
        self.pyramid_depth = len(self.config.fpn_layer_resolutions)
        self.mae = ViTDet(config)

        self.pyramid_layers = []
        for ratio in config.fpn_layer_resolutions:
            if ratio <= 1.0:
                self.pyramid_layers.append(nn.Conv2d(
                    in_channels=config.hidden_size,
                    out_channels=config.fpn_hidden_size,
                    kernel_size=3,
                    stride=int(1.0/ratio),
                    padding=1,
                ))
            elif ratio == 2.0:
                self.pyramid_layers.append(nn.ConvTranspose2d(
                    in_channels=config.hidden_size,
                    out_channels=config.fpn_hidden_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ))
            elif ratio == 4.0:
                self.pyramid_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(
                    in_channels=config.hidden_size,
                    out_channels=config.fpn_hidden_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ), nn.ConvTranspose2d(
                    in_channels=config.fpn_hidden_size,
                    out_channels=config.fpn_hidden_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
                ))
    
    def forward(self, pixel_values: Tensor, gsd: Tensor) -> Mask2FormerBackboneOutput:
        mae_output = self.mae(pixel_values, gsd)
        B, _, S = mae_output.shape
        H, W = self.mae.patch_embedding.patches_resolution
        mae_output = mae_output.reshape(B, H, W, S).permute(0, 3, 1, 2).contiguous() # squarify
        pyramid = []
        for layer in self.pyramid_layers:
            pyramid.append(layer(mae_output))
        
        return Mask2FormerBackboneOutput(pyramid[-1], pyramid, mae_output)

"""
Problems with this backbone:
- My feature pyramid is H/16, H/8, H/4 but the Mask2Former paper claims H/32, H/16, H/8, H/4 (4 feature maps?)
- MAE needs modification to support windowed attention in most layers.
- Dropout to be explored
- Currently can't load checkpoint
"""

import torch

config = ViTDetBackboneConfig(image_size=(512,512), window_size=16)
model = ViTDetBackbone(config)

input = torch.randn(1,3,512,512)
gsd = torch.randn(1)

op = model(input, gsd)
