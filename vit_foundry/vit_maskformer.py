from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from swin import SwinTransformerV2
import math
from vit_components import ACT2FN




@dataclass
class Mask2FormerConfig():
    fpn_hidden_size: int = 256
    transformer_hidden_size: int = 256
    mask_feature_size: int = 256
    num_queries: int = 32
    num_labels: int = 10
    hidden_dim: int = 256
    encoder_feedforward_dim: int = 1024
    activation_function: str = "relu"
    encoder_layers: int = 6
    decoder_layers: int = 10
    num_attention_heads: int = 8
    dropout: float = 0.0
    dim_feedforward: int = 2048
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    use_auxiliary_loss: bool = True
    output_auxiliary_logits: bool = None
    r"""
    This is the configuration class to store the configuration of a [`Mask2FormerModel`]. It is used to instantiate a
    Mask2Former model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Mask2Former
    [facebook/mask2former-swin-small-coco-instance](https://huggingface.co/facebook/mask2former-swin-small-coco-instance)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, Mask2Former only supports the [Swin Transformer](swin) as backbone.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `SwinConfig()`):
            The configuration of the backbone model. If unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        feature_size (`int`, *optional*, defaults to 256):
            The features (channels) of the resulting feature maps.
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
            Dimension of feedforward network for deformable detr encoder used as part of pixel decoder.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of layers in the deformable detr encoder used as part of pixel decoder.
        decoder_layers (`int`, *optional*, defaults to 10):
            Number of layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feature dimension in feedforward network for transformer decoder.
        num_queries (`int`, *optional*, defaults to 100):
            Number of queries for the decoder.
        no_object_weight (`int`, *optional*, defaults to 0.1):
            The weight to apply to the null (no object) class.
        class_weight (`int`, *optional*, defaults to 2.0):
            The weight for the cross entropy loss.
        mask_weight (`int`, *optional*, defaults to 5.0):
            The weight for the mask loss.
        dice_weight (`int`, *optional*, defaults to 5.0):
            The weight for the dice loss.
        train_num_points (`str` or `function`, *optional*, defaults to 12544):
            Number of points used for sampling during loss calculation.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Oversampling parameter used for calculating no. of sampled points
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        use_auxiliary_loss (`boolean``, *optional*, defaults to `True`):
            If `True` [`Mask2FormerForUniversalSegmentationOutput`] will contain the auxiliary losses computed using
            the logits from each decoder's stage.
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.
    """




    

class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


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


class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, hidden_size = 256):
        """
        Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image through a backbone and a pixel
        decoder, generating an image feature map and pixel embeddings.

        Args:
            config ([`MaskFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        self.encoder = SwinTransformerV2()
        self.decoder = MaskFormerPixelDecoder(
            lateral_widths=self.encoder.num_features,
            feature_size=hidden_size,
            mask_feature_size=hidden_size,
        )

    def forward(self, pixel_values: Tensor):
        encoder_features = self.encoder(pixel_values)
        decoder_output = self.decoder(encoder_features)

        return [
            decoder_output[0], # FPN final pixel embeddings 
            decoder_output[1], # FPN feature maps
            encoder_features,  # Swin transformer feature maps
        ]


class Mask2FormerMaskPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor, num_layers: int = 3):
        """
        This class is used to get the predicted mask for a given Mask2FormerMaskedAttentionDecoder layer. It also
        generates the binarized attention mask associated with the given predicted mask. The attention mask obtained
        using predicted mask of the (l-1)th decoder layer is fed to the cross(masked)-attention block of the next
        decoder layer as input.

        Args:
            hidden_size (`int`):
                The feature dimension of the Mask2FormerMaskedAttentionDecoder
            num_heads (`int`):
                The number of heads used in the Mask2FormerMaskedAttentionDecoder
            mask_feature_size (`torch.Tensor`):
                one of the output dimensions of the predicted masks for each query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (num_layers - 1)
        layers.append(nn.Linear(hidden_size, mask_feature_size))
        self.layers = nn.Sequential(*layers)
        ### MAJOR CHANGE: removed some subclasses that just wrapped linear layers


    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: List[int]):
        mask_embeddings = self.layers(outputs)

        # Equivalent to einsum('bqc, bchw -> bqhw') but jit friendly
        '''
        batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        outputs_mask = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
        for c in range(num_channels):
            outputs_mask += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]
        '''
        outputs_mask = torch.einsum('bqc, bchw -> bqhw', mask_embeddings, pixel_embeddings)
        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        # outputs_mask shape:   (batch, seq_length, height, width) (height & width from final FPN encoding)
        # attention_mask shape: (batch * num_heads, seq_length, height * width) (height & width from intermediate encoding)
        return outputs_mask, attention_mask


class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.
    """

    def __init__(
            self,
            hidden_size: int = 256,
            num_attention_heads: int = 8,
            dropout: float = 0.0,
            dim_feedforward: int = 2048,
            activation_fn: str = "relu",
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        ### MAJOR CHANGE: got rid of that custom-implemented self attention module
        self.self_attn = nn.MultiheadAttention(self.hidden_size, num_attention_heads, self.dropout, batch_first=True)
        self.activation_fn = ACT2FN[activation_fn]
        self.activation_dropout = self.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.cross_attn = nn.MultiheadAttention(self.hidden_size, num_attention_heads, self.dropout, batch_first=True)
        self.cross_attn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, self.hidden_size)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_positional_embeddings: Optional[torch.Tensor] = None,
        multi_stage_features: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_positional_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            multi_stage_features (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_positional_embeddings),
            key=self.with_pos_embed(multi_stage_features[level_index], position_embeddings[level_index]),
            value=multi_stage_features[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states

        hidden_states, self_attn_weights = self.self_attn(
            query=self.with_pos_embed(hidden_states, query_positional_embeddings),
            key=self.with_pos_embed(hidden_states, query_positional_embeddings),
            value=hidden_states,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class Mask2FormerMaskedAttentionDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`Mask2FormerMaskedAttentionDecoderLayer`]. The decoder updates the query embeddings through multiple cross
    (masked) and self-attention layers. The decoder uses a new **masked attention** mechanism instead of the standard
    cross-attention, which extracts localized features by constraining cross-attention to within the foreground region
    of the predicted mask for each query, instead of attending to the full feature map.

    Args:
        config (`Mask2FormerConfig`):
            Configuration used to instantiate Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(
            self,
            hidden_size: int = 256,
            decoder_layers: int = 6,
            mask_feature_size: int = 256,
            num_attention_heads: int = 8,
            dropout: float = 0.0):
        super().__init__()

        self.mask_feature_size = mask_feature_size
        self.layerdrop = dropout
        self.num_feature_levels = 3  # level embedding (3 scales)
        self.decoder_layers = decoder_layers - 1

        self.layers = nn.ModuleList(
            [Mask2FormerMaskedAttentionDecoderLayer(hidden_size=hidden_size) for _ in range(self.decoder_layers)]
        )
        self.layernorm = nn.LayerNorm(hidden_size)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor = None,
        query_positional_embeddings: torch.Tensor = None,
        multi_stage_features: torch.Tensor = None,
        multi_stage_positional_embeddings: torch.Tensor = None,
        pixel_embeddings: torch.Tensor = None,
        feature_size_list: List = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Args:
            query_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            query_positional_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            multi_stage_features (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross(masked)-attention of the decoder.
            multi_stage_positional_embeddings (`torch.FloatTensor` of shape `(batch_size, height*width, num_channels)`):
                Position embeddings that are added to the keys in each cross(masked)-attention layer.
            pixel_embeddings (`torch.FloatTensor`):
                Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel
                Decoder.
            feature_size_list (`List[torch.Size]` ):
                This is a list containing shapes (height & width) of multi-scale features from the Pixel Decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """

        hidden_states = query_embeddings

        # intermediate hidden states with layernorm applied - required for predicting class logits
        intermediate_hidden_states = []
        intermediate_mask_predictions = []
        all_hidden_states = []
        attentions = []

        # intermediate mask predictions from transformer decoder layers

        normalized_hidden_states = self.layernorm(hidden_states)
        predicted_mask, attention_mask = self.mask_predictor(
            normalized_hidden_states, pixel_embeddings, feature_size_list[0]
        )
        intermediate_hidden_states.append(normalized_hidden_states)
        intermediate_mask_predictions.append(predicted_mask)

        for idx, decoder_layer in enumerate(self.layers):

            all_hidden_states.append(hidden_states)
            dropout_probability = torch.rand([])
            if self.training and (dropout_probability < self.layerdrop):
                continue
            
            level_index = idx % self.num_feature_levels

            # I don't understand this line... I think it changes any full True layers to False?
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

            layer_outputs = decoder_layer(
                hidden_states,
                level_index=level_index,
                position_embeddings=multi_stage_positional_embeddings,
                query_positional_embeddings=query_positional_embeddings,
                multi_stage_features=multi_stage_features,
                encoder_attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            normalized_hidden_states = self.layernorm(layer_outputs[0])
            predicted_mask, attention_mask = self.mask_predictor(
                normalized_hidden_states,
                pixel_embeddings,
                feature_size_list[(idx + 1) % self.num_feature_levels],
            )
            intermediate_hidden_states.append(normalized_hidden_states)

            intermediate_mask_predictions.append(predicted_mask)
            hidden_states = layer_outputs[0]
            if output_attentions:
                attentions.append(layer_outputs[1])

        # add hidden states from the last decoder layer
        all_hidden_states.append(hidden_states)

        return {
            'hidden_states': hidden_states,
            'all_hidden_states': all_hidden_states if output_hidden_states else None,
            'attentions': attentions if output_attentions else None,
            'intermediate_hidden_states': intermediate_hidden_states, # basically hidden states with a layernorm
            'intermediate_mask_predictions': intermediate_mask_predictions,
        }


class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module.
    """

    def __init__(self, fpn_hidden_size, hidden_size, num_queries):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.num_feature_levels = 3
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        self.query_embeddings = nn.Embedding(self.num_queries, hidden_size)
        self.query_positional_embeddings = nn.Embedding(self.num_queries, hidden_size)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if fpn_hidden_size != hidden_size:
                self.input_projections.append(nn.Conv2d(fpn_hidden_size, hidden_size, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = Mask2FormerMaskedAttentionDecoder(hidden_size=self.hidden_size)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_size)

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        # multi_stage_features are the feature map outputs from the FPN. There are 3 by default.
        # mask_features is the final output of the pixel decoder. Same dimension as the final FPN output, but
        #   has additional transformations
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # Flatten (batch_size, num_channels, height, width) -> (batch_size, height*width, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(0, 2, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(0, 2, 1)

        batch_size, _, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        query_embeddings = self.query_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_positional_embeddings = self.query_positional_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # WHY THE FUCK ARE THEY RENAMING EVERYTHING
        decoder_output = self.decoder(
            query_embeddings=query_embeddings,
            query_positional_embeddings=query_positional_embeddings,
            multi_stage_features=multi_stage_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return decoder_output



def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs
) -> torch.Tensor:
    """
    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg, (1 - labels).T)
    loss = loss_pos + loss_neg
    loss = loss / height_and_width
    return loss


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        """Creates the matcher

        Params:
            cost_class (`float`, *optional*, defaults to 1.0):
                Relative weight of the classification error in the matching cost.
            cost_mask (`float`, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (`float`, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost.
            num_points (`int`, *optional*, defaults to 12544):
                No. of points to sample on which the mask loss will be calculated. The same set of K points are
                uniformly sampled for all prediction and ground truth masks to construct the cost matrix for bipartite
                matching.
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")

        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> List[Tuple[Tensor]]:
        """
        Params:
            masks_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, num_labels` with the classification logits.
            class_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, height, width` with the predicted masks.
            class_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes` (where num_target_boxes is the number of ground-truth objects in the
                target) containing the class labels.
            mask_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes, height, width` containing the target masks.

        Returns:
            matched_indices (`List[Tuple[Tensor]]`): A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            # Sample ground truth and predicted masks
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)

            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)

            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            # final cost matrix
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            # eliminate infinite values in cost_matrix to avoid the error ``ValueError: cost matrix is infeasible``
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return matched_indices


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
class Mask2FormerLoss(nn.Module):
    def __init__(self, config: Mask2FormerConfig, weight_dict: Dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`Mask2FormerConfig`):
                The configuration for Mask2Former model also containing loss calculation specific parameters.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        # Weight to apply to the null class
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=1.0,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # Adapted from nested_tensor_from_tensor_list() in original implementation
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        # compute final size
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # Permute target_classes (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        # Sample point coordinates
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # Permute predictions following indices
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Permute labels following indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the Mask2FormerMaskedAttentionDecoder.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks_pt = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        return num_masks_pt


class Mask2FormerModel(nn.Module):
    def __init__(
            self,
            fpn_hidden_size: int = 256,
            transformer_hidden_size: int = 256,
            num_queries: int = 32,
            ):
        super().__init__()
        self.pixel_level_module = Mask2FormerPixelLevelModule(hidden_size=fpn_hidden_size)
        self.transformer_module = Mask2FormerTransformerModule(
            fpn_hidden_size=fpn_hidden_size,
            hidden_size=transformer_hidden_size,
            num_queries=num_queries,
            )

        #self.post_init()

    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        pixel_level_module_output = self.pixel_level_module(pixel_values=pixel_values)

        transformer_module_output = self.transformer_module(
            mask_features=pixel_level_module_output[0],
            multi_scale_features=pixel_level_module_output[1],
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return {
            'mask_features': pixel_level_module_output[0],
            'multi_scale_features': pixel_level_module_output[1],
            'decoder_output': transformer_module_output,
        }
        '''

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states
        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
        )
        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output
        '''
    



class Mask2FormerForUniversalSegmentation(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.model = Mask2FormerModel(
            fpn_hidden_size=config.fpn_hidden_size,
            transformer_hidden_size=config.transformer_hidden_size,
            num_queries=config.num_queries,
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.transformer_hidden_size, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        #self.post_init()

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        return auxiliary_logits

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs['decoder_output']['intermediate_hidden_states']:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        return {
            'loss': loss,
            'class_queries_logits': class_queries_logits[-1],
            'masks_queries_logits': masks_queries_logits[-1],
            'auxiliary_logits': auxiliary_logits,
            'encoder_last_hidden_state': outputs.encoder_last_hidden_state,
            'pixel_decoder_last_hidden_state': outputs.pixel_decoder_last_hidden_state,
            'transformer_decoder_last_hidden_state': outputs.transformer_decoder_last_hidden_state,
            'encoder_hidden_states': encoder_hidden_states,
            'pixel_decoder_hidden_states': pixel_decoder_hidden_states,
            'transformer_decoder_hidden_states': transformer_decoder_hidden_states,
            'attentions': outputs.attentions,
        }


config = Mask2FormerConfig()
model = Mask2FormerForUniversalSegmentation(config)


ip = torch.randn((1,3,224,224))

op = model(ip, output_hidden_states=True)
print(op['decoder_output']['hidden_states'].shape)
