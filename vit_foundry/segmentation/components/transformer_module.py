
from typing import List, Optional

import torch
from torch import Tensor, nn
from vit_foundry.vit_components import ACT2FN, ViTSinCosPositionalSquareEmbeddings
from vit_foundry.segmentation.config import Mask2FormerTransformerOutput


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
            hidden_size,
            num_attention_heads,
            activation_fn,
            dim_feedforward,
            dropout: float = 0.0,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        ### MAJOR CHANGE: got rid of that custom-implemented self attention module
        self.self_attn = nn.MultiheadAttention(self.hidden_size, num_attention_heads, self.dropout, batch_first=True)
        self.activation_fn = ACT2FN[activation_fn]
        self.activation_dropout = self.dropout

        self.layer_norm_1 = nn.LayerNorm(self.hidden_size)
        self.cross_attn = nn.MultiheadAttention(self.hidden_size, num_attention_heads, self.dropout, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, self.hidden_size)
        self.layer_norm_3 = nn.LayerNorm(self.hidden_size)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_positional_embeddings: Optional[torch.Tensor] = None,
        feature_pyramid: Optional[torch.Tensor] = None,
        feature_pyramid_position_embeddings: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_positional_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            feature_pyramid (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
        """
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_positional_embeddings),
            key=self.with_pos_embed(feature_pyramid, feature_pyramid_position_embeddings),
            value=feature_pyramid,
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm_1(hidden_states)

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
        hidden_states = self.layer_norm_2(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm_3(hidden_states)

        return (hidden_states, self_attn_weights, cross_attn_weights)



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
            hidden_size,
            mask_feature_size,
            decoder_layers,
            num_attention_heads,
            activation_fn,
            dim_feedforward,
            dropout: float = 0.0):
        super().__init__()

        self.mask_feature_size = mask_feature_size
        self.layerdrop = dropout
        self.num_feature_levels = 3  # level embedding (3 scales)
        self.decoder_layers = decoder_layers - 1 # Why is this -1?

        self.layers = nn.ModuleList(
            [Mask2FormerMaskedAttentionDecoderLayer(
                hidden_size,
                num_attention_heads,
                activation_fn,
                dim_feedforward,
            ) for _ in range(self.decoder_layers)]
        )
        # I'm removing this layernorm just to try it out.
        # It's not needed between layers, because the transformer layer end with a layernorm anyway
        # The only question is whether it's useful to use on the original query tokens
        # TODO: test this.
        #self.layernorm = nn.LayerNorm(hidden_size)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor = None,
        query_positional_embeddings: torch.Tensor = None,
        feature_pyramid: torch.Tensor = None,
        feature_pyramid_positional_embeddings: torch.Tensor = None,
        mask_features: torch.Tensor = None,
        feature_size_list: List = None,
    ):
        r"""
        Args:
            query_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            query_positional_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            feature_pyramid (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross(masked)-attention of the decoder.
            feature_pyramid_positional_embeddings (`torch.FloatTensor` of shape `(batch_size, height*width, num_channels)`):
                Position embeddings that are added to the keys in each cross(masked)-attention layer.
            mask_features (`torch.FloatTensor`):
                Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel
                Decoder.
            feature_size_list (`List[torch.Size]` ):
                This is a list containing shapes (height & width) of multi-scale features from the Pixel Decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        hidden_states = [query_embeddings]
        mask_predictions = []
        attentions = []
        
        predicted_mask, attention_mask = self.mask_predictor(
            hidden_states[-1], mask_features, feature_size_list[0]
        )
        mask_predictions.append(predicted_mask)

        for idx, decoder_layer in enumerate(self.layers):

            dropout_probability = torch.rand([])
            if self.training and (dropout_probability < self.layerdrop):
                continue
            
            level_index = idx % self.num_feature_levels

            # I don't understand this line... I think it changes any full True layers to False?
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

            layer_outputs = decoder_layer(
                hidden_states[-1],
                query_positional_embeddings=query_positional_embeddings,
                feature_pyramid=feature_pyramid[level_index],
                feature_pyramid_position_embeddings=feature_pyramid_positional_embeddings[level_index],
                encoder_attention_mask=attention_mask,
            )
            hidden_states.append(layer_outputs[0])
            attentions.append(layer_outputs[1])

            predicted_mask, attention_mask = self.mask_predictor(
                hidden_states[-1],
                mask_features,
                feature_size_list[(idx + 1) % self.num_feature_levels],
            )
            mask_predictions.append(predicted_mask)


        return {
            'hidden_states': hidden_states,
            'mask_predictions': mask_predictions,
            'attentions': attentions,
        }


class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module. See config for parameter descriptions.
    """

    def __init__(
            self,
            hidden_size = 256,
            fpn_hidden_size = 256,
            mask_feature_size = 256,
            num_queries = 100,
            num_feature_levels = 3,
            decoder_layers = 6,
            num_attention_heads = 8,
            dim_feedforward = 2048,
            activation_fn = 'relu',
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels

        self.position_embedder = ViTSinCosPositionalSquareEmbeddings()
        self.query_embeddings = nn.Embedding(num_queries, hidden_size)
        self.query_positional_embeddings = nn.Embedding(num_queries, hidden_size)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if fpn_hidden_size != hidden_size:
                self.input_projections.append(nn.Conv2d(fpn_hidden_size, hidden_size, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = Mask2FormerMaskedAttentionDecoder(
            hidden_size,
            mask_feature_size,
            decoder_layers * num_feature_levels,
            num_attention_heads,
            activation_fn,
            dim_feedforward,
        )
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_size)

    def forward(
            self,
            mask_features: Tensor,
            feature_pyramid: List[Tensor],
    ) -> Mask2FormerTransformerOutput:
        feature_pyramid_with_level_embedding = []
        feature_pyramid_positional_embeddings = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(feature_pyramid[i].shape[-2:])
            feature_pyramid_positional_embeddings.append(self.position_embedder(feature_pyramid[i]).flatten(2))
            feature_pyramid_with_level_embedding.append(
                self.input_projections[i](feature_pyramid[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # Flatten (batch_size, num_channels, height, width) -> (batch_size, height*width, num_channels)
            feature_pyramid_positional_embeddings[-1] = feature_pyramid_positional_embeddings[-1].permute(0, 2, 1)
            feature_pyramid_with_level_embedding[-1] = feature_pyramid_with_level_embedding[-1].permute(0, 2, 1)

        batch_size, _, _ = feature_pyramid_with_level_embedding[0].shape

        # [num_queries, batch_size, num_channels]
        query_embeddings = self.query_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_positional_embeddings = self.query_positional_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        decoder_output = self.decoder(
            query_embeddings=query_embeddings,
            query_positional_embeddings=query_positional_embeddings,
            feature_pyramid=feature_pyramid_with_level_embedding,
            feature_pyramid_positional_embeddings=feature_pyramid_positional_embeddings,
            mask_features=mask_features,
            feature_size_list=size_list,
        )

        return Mask2FormerTransformerOutput(**decoder_output)


# Problems with this transformer module
# - Layernorm on original tokens to be explored
# - Dropout to be explored
# - Initialization to be explored
