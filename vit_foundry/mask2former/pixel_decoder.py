
from typing import Dict, List, Optional, Tuple

from torch import Tensor, nn

from vit_foundry.mask2former.config import Mask2FormerSinePositionEmbedding


class MaskFormerFPNConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1):
        """
        A basic module that executes conv - norm - in sequence used in MaskFormer.

        Args:
            in_features (`int`):
                The number of input features (channels).
            out_features (`int`):
                The number of outputs features (channels).
        """
        super().__init__()
        self.layers = [
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(32, out_features),
            nn.ReLU(inplace=True),
        ]
        for i, layer in enumerate(self.layers):
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class MaskFormerFPNLayer(nn.Module):
    def __init__(self, in_features: int, lateral_features: int):
        """
        A Feature Pyramid Network Layer (FPN) layer. It creates a feature map by aggregating features from the previous
        and backbone layer. Due to the spatial mismatch, the tensor coming from the previous layer is upsampled.

        Args:
            in_features (`int`):
                The number of input features (channels).
            lateral_features (`int`):
                The number of lateral features (channels).
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(lateral_features, in_features, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(32, in_features),
        )

        self.block = MaskFormerFPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = nn.functional.interpolate(down, size=left.shape[-2:], mode="nearest")
        down += left
        down = self.block(down)
        return down

    
class MaskFormerPixelDecoder(nn.Module):
    def __init__(self, lateral_widths: List[int], feature_size: int = 256, mask_feature_size: int = 256):
        r"""
        Pixel Decoder Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It first runs the backbone's features into a Feature Pyramid
        Network creating a list of feature maps. Then, it projects the last one to the correct `mask_size`.

        Args:
            lateral_widths (`List[int]`):
                Set of hidden sizes for the various feature layers in the encoder. ex [96, 192, 384, 768]
            backbone_feature_size (`int`, *optional*, defaults to 256):
                The feature size (channel dimension) of the final feature map from the backbone.
            mask_feature_size (`int`, *optional*, defaults to 256):
                The features (channels) of the target masks size \\(C_{\epsilon}\\) in the paper.
        """
        super().__init__()
        encoder_final_width = lateral_widths[-1]
        encoder_medial_widths = lateral_widths[:-1]
        self.stem = MaskFormerFPNConvLayer(encoder_final_width, feature_size)
        
        self.layers = nn.Sequential(
            *[MaskFormerFPNLayer(feature_size, lateral_width) for lateral_width in encoder_medial_widths[::-1]]
        )
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=3, padding=1)
        num_pos_feats = feature_size // 2
        self.position_embedding = Mask2FormerSinePositionEmbedding(num_pos_feats=num_pos_feats, normalize=True)

    def forward(self, backbone_features):
        fpn_features = []
        last_feature = backbone_features[-1]
        other_features = backbone_features[:-1]
        output = self.stem(last_feature)
        for layer, left in zip(self.layers, other_features[::-1]):
            output = layer(output, left)
            fpn_features.append(self.position_embedding(output))


        last_feature_projected = self.mask_projection(fpn_features[-1])

        return (last_feature_projected, tuple(fpn_features))
