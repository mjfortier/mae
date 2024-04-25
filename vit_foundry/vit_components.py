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
    cls_token: bool = False


########################
# Embedding  and Masking
########################

class ViTMAEPatchEmbeddings(nn.Module):
    def __init__(self, config, norm_layer=None):
        super().__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patches_resolution = [self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.projection = nn.Conv2d(self.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = None
        self.initialize_weights()

    def initialize_weights(self):
        self.projection.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.projection.bias.data.zero_()

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size != 0:
            pad_values = (0, self.patch_size - width % self.patch_size)
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size != 0:
            pad_values = (0, 0, 0, self.patch_size - height % self.patch_size)
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values):
        '''
        (in)  pixel_values - (B,N,H,W)
        (out) embeddings - (B,S,H)
        '''
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        if self.norm is not None:
            embeddings = self.norm(embeddings)
        return embeddings


class ViTSinCosPositionalEmbeddings(nn.Module):
    '''
    This module takes input of shape (Batch, seq_len, hidden_size)
    Advantage: May be slightly more performant as some parameters are cached
    Disadvantage: Requires knowledge of image parameters at initialization; less flexible
    '''
    def __init__(self, image_size, patch_size, hidden_size, temperature: int = 10000):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.num_patches_y = self.image_size[0] // self.patch_size
        self.num_patches_x = self.image_size[1] // self.patch_size
        self.num_patches = self.num_patches_y * self.num_patches_x

        self.position_embeddings_block_x = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_size // 4), requires_grad=False
        )
        self.position_embeddings_block_y = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_size // 4), requires_grad=False
        )
        self.initialize_weights()

    def initialize_weights(self):
        assert self.hidden_size % 4 == 0, "embed_dim must be divisible by 4"
        grid_h = np.arange(self.num_patches_y, dtype=float)

        embed_dim = self.hidden_size // 4 # sin, cos, x, y
        omega = np.arange(embed_dim, dtype=float) / embed_dim
        omega = 1.0 / self.temperature**omega
        block = np.einsum("m,d->md", grid_h, omega)
        block_x = np.concatenate([block] * self.num_patches_x, axis=0)
        block_y = np.repeat(block, self.num_patches_x, axis=0)

        self.position_embeddings_block_x.data.copy_(torch.from_numpy(block_x).float().unsqueeze(0))
        self.position_embeddings_block_y.data.copy_(torch.from_numpy(block_y).float().unsqueeze(0))


    def forward(self, tokens, scaling = None):
        '''
        (in)  embeddings - (B,L,S)
              gsd - (B)
        (out) embeddings - (B,L,S)
        '''
        # Assumes no CLS token
        B, _, _ = tokens.shape
        x_embed = self.position_embeddings_block_x.repeat(B, 1, 1)
        y_embed = self.position_embeddings_block_y.repeat(B, 1, 1)

        if scaling is not None:
            scaling = scaling.reshape(B, 1, 1)
            x_embed *= scaling
            y_embed *= scaling

        emb_sin_x = torch.sin(x_embed) # (B, S, H/4)
        emb_cos_x = torch.cos(x_embed)
        emb_sin_y = torch.sin(y_embed)
        emb_cos_y = torch.cos(y_embed)
        positional_embedding = torch.cat([emb_sin_x, emb_cos_x, emb_sin_y, emb_cos_y], dim=2)
        return tokens + positional_embedding
    
    
class ViTSinCosPositionalSquareEmbeddings(nn.Module):
    '''
    This module takes input of shape (Batch, hidden_size, H, W)
    Advantage: Image size agnostic, and keeps patches in rectangular shape
    Disadvantage: More computation on-the-fly as the embeddings are recalculated
    '''
    def __init__(self, temperature: int = 10000):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, scaling: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, H, W = x.shape
        grid = torch.ones((B, H, W), device=x.device, dtype=x.dtype)
        y_embed = grid.cumsum(1) - 1.0
        x_embed = grid.cumsum(2) - 1.0
        if scaling is not None:
            scaling = scaling.reshape(B, 1, 1)
            x_embed *= scaling
            y_embed *= scaling

        dim_t = torch.arange(S / 4, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (4 * dim_t / S)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos = torch.cat((pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()), dim=3).permute(0, 3, 1, 2)
    
    
class ViTSinCos1DPositionalEmbeddings(nn.Module):
    '''
    This module takes input of shape (Batch, hidden_size, H, W)
    Advantage: Image size agnostic, and keeps patches in rectangular shape
    Disadvantage: More computation on-the-fly as the embeddings are recalculated
    '''
    def __init__(self, hidden_size, temperature: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        inv_freq = 1.0 / (temperature ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == x.shape:
            return self.cached_penc

        self.cached_penc = None
        B, L, H = x.shape
        pos_x = torch.arange(L, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((L, self.hidden_size), device=x.device, dtype=x.dtype)
        emb[:, : self.hidden_size] = emb_x

        self.cached_penc = emb[None, :, :H].repeat(B, 1, 1)
        return self.cached_penc


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
        if ids_restore.shape[1] % 2 == 0:
            # no CLS token
            mask_tokens = self.mask_token.repeat(h.shape[0], ids_restore.shape[1] - h.shape[1], 1)
            h = torch.cat([h, mask_tokens], dim=1)
            h = torch.gather(h, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, h.shape[2]))  # unshuffle
        else:
            # with CLS token
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
            mask = mask.to(sequence.device)
            len_keep = torch.where(mask==0)[0].size()[0]
            if mask.shape[0] < batch_size:
                mask = mask.repeat(batch_size, 1)
            
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

class MultiheadAttentionBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            kv_hidden_size=None,
            qkv_bias=True,
            attention_dropout_prob=0.0,
            hidden_dropout_prob=0.0
            )-> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"The hidden size {hidden_size} is not a multiple of the number of attention heads ({num_heads}).")

        self.hidden_size = hidden_size
        self.kv_hidden_size = hidden_size if kv_hidden_size is None else kv_hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.key = nn.Linear(self.kv_hidden_size, hidden_size, bias=qkv_bias)
        self.value = nn.Linear(self.kv_hidden_size, hidden_size, bias=qkv_bias)

        self.attention_dropout = nn.Dropout(attention_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            q_states: torch.Tensor,
            kv_states: Optional[torch.Tensor],
            mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        '''
        hidden_states - (B, L, H)
        cross_attention_states - (B, Lc, Hc)
        mask: (B, L, Lc)
            - where mask is True, attention score will be set to '-inf'
        '''
        
        query_layer = self.transpose_for_scores(self.query(q_states)) # (B, num_heads, L, head_size)
        key_layer   = self.transpose_for_scores(self.key(kv_states)) # (B, num_heads, Lc, head_size)
        value_layer = self.transpose_for_scores(self.value(kv_states)) # (B, num_heads, Lc, head_size)
    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (B, num_heads, L, Lc)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.where(~mask, float('-inf'))
        attention_probs = self.attention_dropout(attention_scores) # NOTE: This was in the original paper, but it drops entire tokens so we could try removing it
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)


        context_layer = torch.matmul(attention_probs, value_layer) # (B, num_heads, L, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape) # (B, L, H)

        output = self.dense(context_layer)
        output = self.hidden_dropout(output)
        
        return (output, attention_probs)


class AttentionLayer(nn.Module):
    '''
    Basic (pre-LN) attention layer:
        - Layernorm & MHA
        - residual
        - Layernorm & FFN
        - residual
    '''
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio,
            kv_hidden_size=None,
            qkv_bias=True,
            activation='gelu', # can also pass a nn.Module
            attention_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
            eps=1e-12,
            )-> None:
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = MultiheadAttentionBlock(hidden_size, num_heads, kv_hidden_size=kv_hidden_size, qkv_bias=qkv_bias,
                                                 attention_dropout_prob=attention_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)
        
        self.layernorm_2 = nn.LayerNorm(hidden_size, eps=eps)
    
        if isinstance(activation, str):
            activation = ACT2FN[activation]
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            activation,
            nn.Linear(mlp_ratio * hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        q_states: torch.Tensor,
        kv_states: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        if kv_states is None:
            kv_states = q_states
        
        residual = q_states
        h = self.layernorm_1(q_states)
        h, att = self.attention(h, kv_states, mask=mask)
        h = h + residual

        residual = h
        h = self.layernorm_2(h)
        h = self.ffn(h)
        h = self.dropout(h)
        h = h + residual
        
        if output_attentions:
            return (h, att)
        else:
            return (h, None)


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
        cross_attention_states: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:

        if cross_attention_states is None:
            cross_attention_states = hidden_states # self attention
        
        # QKV operations
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer   = self.transpose_for_scores(self.key(cross_attention_states))
        value_layer = self.transpose_for_scores(self.value(cross_attention_states))
    
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


class ViTMAECrossAttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layernorm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = ViTMAEAttentionBlock(config)
        self.layernorm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attention = ViTMAEAttentionBlock(config)
        self.layernorm_3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        cross_attention_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        
        h = self.layernorm_1(hidden_states)
        h, att1 = self.attention(h, head_mask, output_attentions)
        h = h + hidden_states
        
        intermediate = h
        h = self.layernorm_2(hidden_states)
        h, att2 = self.cross_attention(
            h,
            cross_attention_states=cross_attention_states,
            head_mask=head_mask,
            output_attentions=output_attentions)
        h = h + intermediate
        
        intermediate = h
        h = self.layernorm_3(h)
        h = self.fc1(h)
        h = self.fc_act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = h + intermediate
        
        if output_attentions:
            return (h, [att1, att2])
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


class ViTEncoderDecoder(nn.Module):
    '''
    Parallel encoder and decoder modules, where the decoder constantly gets cross-attention
    from the encoder layers.
    '''
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList([ViTMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.decoder_layers = nn.ModuleList([ViTMAECrossAttentionLayer(config) for _ in range(config.num_hidden_layers)])
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
        enc_hidden_states: torch.Tensor,
        dec_hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        
        enc_attentions = []
        dec_attentions = []
        for enc_module, dec_module in zip(self.encoder_layers, self.decoder_layers):
            enc_hidden_states, enc_att = enc_module(enc_hidden_states, head_mask, output_attentions)
            dec_hidden_states, dec_att = dec_module(dec_hidden_states, enc_hidden_states, head_mask, output_attentions)
            enc_attentions.append(enc_att)
            dec_attentions.append(dec_att)

        if output_attentions:
            return (enc_hidden_states, dec_hidden_states, [enc_attentions, dec_attentions])
        else:
            return (enc_hidden_states, dec_hidden_states, None)


