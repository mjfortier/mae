from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import Tensor, nn
from vit_foundry.segmentation.components.swin import SwinTransformerV2
from vit_foundry.segmentation.components.fpn import FeaturePyramidNetwork
from vit_foundry.segmentation.config import Mask2FormerBackboneOutput


@dataclass
class SwinFPNBackboneConfig():
    # Swin parameters
    img_size: Tuple[int] = (224,224)
    window_size: int = 7                 # pixels per window edge
    backbone_embed_dim: int = 128        # hidden size per token. Note: this doubles every layer in Swin transformers.
    depths: List[int] = (2, 2, 18)       # swin blocks per layer
    num_heads: List[int] = (4, 8, 16)    # attention heads per layer
    drop_path_rate: float = 0.2          # dropout of some kind I can't remember at time of writing

    # FPN parameters
    fpn_hidden_size: int = 256           # all feature pyramid entries will have this size
    mask_feature_size: int = 256         # final projection size of feature mask (1x1 conv from last feature pyramid)


class SwinFPNBackbone(nn.Module):
    def __init__(self, config: SwinFPNBackboneConfig, checkpoint: str = None):
        super().__init__()
        self.config = config
        self.pyramid_depth = len(self.config.depths)
        self.swin = SwinTransformerV2(
            img_size=config.img_size,
            window_size=config.window_size,
            embed_dim=config.backbone_embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
        )
        self.fpn = FeaturePyramidNetwork(
            lateral_widths=self.swin.num_features,
            feature_size=config.fpn_hidden_size,
            mask_feature_size=config.mask_feature_size,
        )

        if checkpoint:
            self.load_checkpoint(checkpoint, strict=False)


    def load_checkpoint(self, checkpoint):
        state_dict = torch.load(checkpoint)
        self.swin.load_state_dict(state_dict['model'])


    def forward(self, pixel_values: Tensor) -> Mask2FormerBackboneOutput:
        swin_features = self.swin(pixel_values)
        fpn_features = self.fpn(swin_features)
        return Mask2FormerBackboneOutput(fpn_features[0], fpn_features[1], swin_features)

"""
Problems with this backbone:
- My feature pyramid is H/16, H/8, H/4 but the Mask2Former paper claims H/32, H/16, H/8
- Dropout to be explored
- Initialization to be explored
- Checkpoint loading is dicey; can silently fail
"""


checkpoint = '/home/matt/projects/core_research/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'

config = SwinFPNBackboneConfig(img_size=(384,384), window_size=24)
model = SwinFPNBackbone(config, checkpoint)


