import numpy as np
import math
import torch
from torch import nn
from typing import Optional, Set, Tuple, Union
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(input)

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


@dataclass
class ViTMAEConfig():
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    z: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: int = 1e-12
    image_size: Tuple[int] = (256,256)
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    zc: int = 16
    decoder_hidden_size: int = 512
    decoder_num_hidden_layers: int = 8
    decoder_num_attention_heads: int = 16
    decoder_mlp_ratio: int = 4
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False


########################
# Embedding  and Masking
########################

class ViTMAEPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size[1] // self.patch_size) * (self.image_size[0] // self.patch_size)

        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        self.projection.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.projection.bias.data.zero_()

    def forward(self, pixel_values):
        '''
        (in)  pixel_values - (B,N,H,W)
        (out) x - (B,S,H)
        '''
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class ViTMAEPositionalEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches_y = self.image_size[0] // self.patch_size
        self.num_patches_x = self.image_size[1] // self.patch_size
        self.num_patches = self.num_patches_y * self.num_patches_x

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_size), requires_grad=False
        )
        self.initialize_weights()

    def initialize_weights(self):
        assert self.hidden_size % 4 == 0, "embed_dim must be divisible by 4"
        grid_h = np.arange(self.num_patches_y, dtype=np.float32)
        grid_w = np.arange(self.num_patches_x, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, self.num_patches_y, self.num_patches_x])

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(grid[1])  # (H*W, D/2)
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

        self.position_embeddings.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))

    def get_1d_sincos_pos_embed_from_grid(self, pos):
        embed_dim = self.hidden_size // 2

        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def forward(self, embeddings):
        '''
        (in)  embeddings - (B,S,H)
        (out) embeddings - (B,S,H)
        '''
        # check for CLS token - don't need to add positional embedding to it
        if embeddings.shape[1] == self.num_patches + 1:
            embeddings[:,1:] = embeddings[:,1:] + self.position_embeddings
        else:
            embeddings = embeddings + self.position_embeddings
        return embeddings


class ViTMAERandomMasking(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def add_mask_tokens_and_unshuffle(self, h, ids_restore):
        '''
        NOTE: The mask token dimension is taken from the config decoder hidden size. This is
              because tokens are projected to the decoder hidden size before unshuffling.
        '''
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(h.shape[0], ids_restore.shape[1] + 1 - h.shape[1], 1)
        h_ = torch.cat([h[:, 1:, :], mask_tokens], dim=1)  # no cls token
        h_ = torch.gather(h_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, h.shape[2]))  # unshuffle
        h = torch.cat([h[:, :1, :], h_], dim=1)  # append cls token
        return h

    def forward(self, sequence, mask=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape

        if mask is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

            len_keep = int(seq_length * (1 - self.config.mask_ratio))
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([batch_size, seq_length], device=sequence.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            len_keep = torch.where(mask==0)[0].size()[0]
            # sort noise for each sample
            ids_shuffle = torch.argsort(mask, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        return sequence_unmasked, mask, ids_restore


###############################
# Computational Components
###############################

class ViTMAEAttentionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.attention_probs_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:

        # QKV operations
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer   = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) # NOTE: this happens AFTER masking in the original paper
        attention_probs = self.attention_probs_dropout(attention_probs) # NOTE: This was in the original paper, but it drops entire tokens so we could try removing it

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        output = self.dense(context_layer) # NOTE: feels like this shouldn't work d/t size constraints... test?
        output = self.hidden_dropout(output)
        
        if output_attentions:
            return (output, attention_probs)
        else:
            return (output, None)


class ViTMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layernorm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = ViTMAEAttentionBlock(config)
        self.layernorm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.fc1 = nn.Linear(config.hidden_size, config.mlp_ratio * config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.fc_act = ACT2FN[config.hidden_act]
        else:
            self.fc_act = config.hidden_act
        self.fc2 = nn.Linear(config.mlp_ratio * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        
        h = self.layernorm_1(hidden_states)
        h, att = self.attention(h, head_mask, output_attentions)
        h = h + hidden_states
        intermediate = h

        h = self.layernorm_2(h)
        h = self.fc1(h)
        h = self.fc_act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = h + intermediate
        
        if output_attentions:
            return (h, att)
        else:
            return (h, None)


class ViTMAEEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ViTMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        
        attentions = []
        for layer_module in self.layers:
            hidden_states, att = layer_module(hidden_states, head_mask, output_attentions)
            attentions.append(att)

        if output_attentions:
            return (hidden_states, attentions)
        else:
            return (hidden_states, None)


class ViTMAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.mlp_ratio = config.decoder_mlp_ratio
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )
        self.config = config
        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions=False,
    ):

        attentions = []
        for layer_module in self.decoder_layers:
            hidden_states, att = layer_module(hidden_states, head_mask, output_attentions)
            attentions.append(att)
        
        if output_attentions:
            return (hidden_states, attentions)
        else:
            return (hidden_states, None)



