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
    window_size: int = 7                 # pixels per window edge
    backbone_embed_dim: int = 128        # hidden size per token. Note: this doubles every layer in Swin transformers.
    depths: List[int] = (2, 2, 18, 2)       # swin blocks per layer
    num_heads: List[int] = (4, 8, 16, 32)    # attention heads per layer
    drop_path_rate: float = 0.3          # dropout of some kind I can't remember at time of writing

    # FPN parameters
    fpn_hidden_size: int = 256           # all feature pyramid entries will have this size
    mask_feature_size: int = 256         # final projection size of feature mask (1x1 conv from last feature pyramid)


class SwinFPNBackbone(nn.Module):
    def __init__(self, config: SwinFPNBackboneConfig, checkpoint: str = None):
        super().__init__()
        self.config = config
        self.pyramid_depth = len(self.config.depths)
        self.swin = SwinTransformerV2(
            window_size=config.window_size,
            embed_dim=config.backbone_embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate
        )
        self.fpn = FeaturePyramidNetwork(
            lateral_widths=self.swin.num_features,
            feature_size=config.fpn_hidden_size,
            mask_feature_size=config.mask_feature_size,
        )

        if checkpoint:
            self.load_checkpoint(checkpoint)


    def load_checkpoint(self, checkpoint):
        state_dict = torch.load(checkpoint)
        renamed_state_dict = {}
        for key in state_dict['state_dict'].keys():
            if 'backbone' in key:
                renamed = key.removeprefix('backbone.')
                renamed_state_dict[renamed] = state_dict['state_dict'][key]
        self.swin.load_state_dict(renamed_state_dict, strict=False)


    def forward(self, pixel_values: Tensor) -> Mask2FormerBackboneOutput:
        swin_features = self.swin(pixel_values)
        fpn_features = self.fpn(swin_features)
        return Mask2FormerBackboneOutput(fpn_features[0], fpn_features[1], swin_features)


checkpoint = '/home/matt/projects/core_research/upernet_swin_base_patch4_window7_512x512.pth'
import numpy as np
config = SwinFPNBackboneConfig(window_size=7)
model = SwinFPNBackbone(config, checkpoint)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
ip = torch.randn(1,3,512,512)
op = model(ip)
print([f.shape for f in op.encoder_features])
print([f.shape for f in op.feature_pyramid])
print(op.mask_features.shape)

print(model.swin.layers[3].downsample.norm.weight)
print(model.swin.layers[3].downsample.norm.bias)
print(model.swin.layers[3].downsample.reduction.weight)
