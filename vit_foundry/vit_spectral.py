import torch
import torch.nn as nn


class SpectralPatchEmbedding(nn.Module):
    def __init__(self, patch_size, spectral_hidden_size, pixel_hidden_size, token_hidden_size, max_bands=12):
        super().__init__()
        self.patch_size = patch_size
        self.pixel_hidden_size = pixel_hidden_size
        self.spectral_hidden_size = spectral_hidden_size
        self.token_hidden_size = token_hidden_size
        self.pixel_projection = nn.Conv2d(1, pixel_hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.band_projection = nn.Embedding(max_bands, self.spectral_hidden_size)
        self.hidden_projection = nn.Linear(self.pixel_hidden_size + self.spectral_hidden_size, self.token_hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        self.pixel_projection.weight.data.normal_(mean=0.0, std=0.02)
        self.pixel_projection.bias.data.zero_()
        self.hidden_projection.weight.data.normal_(mean=0.0, std=0.02)
        self.hidden_projection.bias.data.zero_()

    def forward(self,
                pixel_values: torch.Tensor,
                band_values: torch.Tensor,
                ):
        '''
        (in) pixel_values - (B, C, H, W)
             band_values - (B, C)
        '''
        B, C, H, W = pixel_values.shape
        Ht = H // self.patch_size
        Wt = W // self.patch_size
        h = pixel_values.reshape([B*C, 1, H, W])
        h = self.pixel_projection(h).reshape(B, C, self.pixel_hidden_size, Ht * Wt)

        b = band_values.flatten()
        b = self.band_projection(b).reshape(B, C, self.spectral_hidden_size)
        b = b.unsqueeze(-1).repeat(1, 1, 1, Ht * Wt)

        h = torch.cat([h, b], dim=2).permute((0, 1, 3, 2)) # B, C, L, H
        h = self.hidden_projection(h).permute((0, 2, 1, 3)) # B, L, C, H

        h = h.sum(dim=2) # B, L, H


ip_pixels = torch.randn(16, 6, 384, 384)
ip_bands = torch.randint(12, (16,6))
model = SpectralPatchEmbedding(16, 32, 128, 256)
model.forward(ip_pixels, ip_bands)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))