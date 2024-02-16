import torch
import numpy as np
from torch import nn
import vit_foundry.vit_components as vc

class GeospatialMAEPositionalEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
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
        omega = 1.0 / 10000**omega
        block = np.einsum("m,d->md", grid_h, omega)
        block_x = np.concatenate([block] * self.num_patches_x, axis=0)
        block_y = np.repeat(block, self.num_patches_x, axis=0)

        self.position_embeddings_block_x.data.copy_(torch.from_numpy(block_x).float().unsqueeze(0))
        self.position_embeddings_block_y.data.copy_(torch.from_numpy(block_y).float().unsqueeze(0))


    def forward(self, tokens, gsd):
        '''
        (in)  embeddings - (B,L,S)
              gsd - (B)
        (out) embeddings - (B,L,S)
        '''
        # Assumes no CLS token
        B, _, _ = tokens.shape
        emb_x = self.position_embeddings_block_x.repeat(B,1,1)
        emb_y = self.position_embeddings_block_y.repeat(B,1,1)

        gsd = gsd.unsqueeze(1).unsqueeze(2) # (B, 1, 1)

        emb_sin_x = torch.sin(gsd * emb_x) # (B, S, H/4)
        emb_cos_x = torch.cos(gsd * emb_x)
        emb_sin_y = torch.sin(gsd * emb_y)
        emb_cos_y = torch.cos(gsd * emb_y)
        positional_embedding = torch.cat([emb_sin_x, emb_cos_x, emb_sin_y, emb_cos_y], dim=2)
        return tokens + positional_embedding


class ViTDetEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([vc.ViTMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.global_attention_layer_interval = config.global_attention_layer_interval
        self.window_size = config.window_size
        self.patch_res = (config.image_size[0] // config.window_size, config.image_size[1] // config.window_size)
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


    def window_partition(self, x):
        """
            IN: x - (B, L, S)
            OUT: windows - (B*ws*ws, L/(ws*ws)), S)
        """
        B, L, S = x.shape
        H, W = self.patch_res
        ws = self.window_size
        x = x.view(B, L//H, L//W, S)
        x = x.view(B, H//ws, ws, W//ws, ws, S)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, ws*ws, S)
        return windows


    def window_reverse(self, windows):
        """
        IN: windows - (B*ws*ws, L/(ws*ws)), S)
        OUT: x - (B, L, S)
        """
        H, W = self.patch_res
        ws = self.window_size
        B = windows.shape[0] // ((H//ws)*(W//ws))
        x = windows.view(B, H//ws, W//ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H*W, -1)
        return x


    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        
        attentions = []
        hidden_states = self.window_partition(hidden_states)

        for i, layer_module in enumerate(self.layers):
            if (i+1)%self.global_attention_layer_interval == 0:

                hidden_states = self.window_reverse(hidden_states)
                hidden_states, att = layer_module(hidden_states, head_mask, output_attentions)
                attentions.append(att)
                hidden_states = self.window_partition(hidden_states)

            else:
                hidden_states, att = layer_module(hidden_states, head_mask, output_attentions)
                attentions.append(att)
            
        hidden_states = self.window_reverse(hidden_states)
        if output_attentions:
            return (hidden_states, attentions)
        else:
            return (hidden_states, None)


class ViTDet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = vc.ViTMAEPatchEmbeddings(config)
        self.enc_positional_embedding = GeospatialMAEPositionalEmbeddings(config.image_size, config.patch_size, config.hidden_size)
        self.encoder = ViTDetEncoder(config)

    def forward(self, pixel_values, gsd, output_attentions: bool = False):
        h = self.patch_embedding(pixel_values)
        h = self.enc_positional_embedding(h, gsd)

        # encoder
        h, _ = self.encoder(h, output_attentions=output_attentions)

        return h
