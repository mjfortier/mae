import torch
import numpy as np
from torch import nn
import vit_foundry.vit_components as vc


class ViTDetEncoder(nn.Module):
    '''
    This mimics a ViTMAE, but adds in windowing for most layers.
    TODO: Make it ingest a checkpoint from a pretrained ViT
    '''
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
        self.enc_positional_embedding = vc.ViTSinCosPositionalEmbeddings(config.image_size, config.patch_size, config.hidden_size)
        self.encoder = ViTDetEncoder(config)

    def forward(self, pixel_values, gsd, output_attentions: bool = False):
        h = self.patch_embedding(pixel_values)
        h = self.enc_positional_embedding(h, gsd)

        # encoder
        h, _ = self.encoder(h, output_attentions=output_attentions)

        return h
