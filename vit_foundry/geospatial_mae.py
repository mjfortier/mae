import os
import torch
import numpy as np
from torch import nn
import vit_foundry.vit_components as vc
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json


def vit_mae_base_config():
    return vc.ViTMAEConfig(
        hidden_size = 768,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        decoder_hidden_size = 512,
        decoder_num_hidden_layers = 8,
        decoder_num_attention_heads = 16
    )

def vit_mae_large_config():
    return vc.ViTMAEConfig(
        hidden_size = 1024,
        num_hidden_layers = 24,
        num_attention_heads = 16,
        decoder_hidden_size = 512,
        decoder_num_hidden_layers = 8,
        decoder_num_attention_heads = 16
    )

def vit_mae_huge_config():
    return vc.ViTMAEConfig(
        hidden_size = 1280,
        num_hidden_layers = 32,
        num_attention_heads = 16,
        decoder_hidden_size = 512,
        decoder_num_hidden_layers = 8,
        decoder_num_attention_heads = 16
    )


# Custom image dataset; pytorch probably has something identical to this already, but
# I'm just quickly throwing this in as a 100%-compatible dataset for dataloader.
class GeospatialMAEDataset(Dataset):
    def __init__(self, root_dirs, split="train", image_size=(224,224)):
        """
        Args:
            root_dirs (list[string]): One or more root dir paths.
            split (string): Either "train" or "val"
        """
        self.root_dirs = root_dirs
        self.split = split
        self.transform = transforms.ToTensor()
        self.images = []
        self.image_size = image_size
        
        for dir in self.root_dirs:
            with open(os.path.join(dir, f'{self.split}.json'), 'r') as f:
                ds_meta = json.load(f)
                ds_meta = [(os.path.join(dir, e['filename']), e) for e in ds_meta if e['height'] >= self.image_size[0] and e['width'] >= self.image_size[1]]
                self.images.extend(ds_meta)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename, meta = self.images[idx]
        image = Image.open(filename)
        image = self.transform(image)

        # Random crop
        crop_h, crop_w = self.image_size
        h = meta['height']
        w = meta['width']
        offset_h = np.random.randint(h - crop_h)
        offset_w = np.random.randint(w - crop_w)
        image = image[:, offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]

        return (image, torch.tensor(meta['GSD']))




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
        (in)  embeddings - (B,S,H)
              gsd - (B)
        (out) embeddings - (B,S,H)
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



class GeospatialMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = vc.ViTMAEPatchEmbeddings(config)
        self.enc_positional_embedding = GeospatialMAEPositionalEmbeddings(config.image_size, config.patch_size, config.hidden_size)
        self.random_masking = vc.ViTMAERandomMasking(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.encoder = vc.ViTMAEEncoder(config)
        self.enc_dec_projection = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.dec_positional_embedding = GeospatialMAEPositionalEmbeddings(config.image_size, config.patch_size, config.decoder_hidden_size)
        self.decoder = vc.ViTMAEDecoder(config)
        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size * config.patch_size * config.num_channels, bias=True
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
        torch.nn.init.normal_(self.enc_dec_projection.weight.data, std=self.config.initializer_range)
        torch.nn.init.normal_(self.decoder_pred.weight.data, std=self.config.initializer_range)
        torch.nn.init.ones_(self.decoder_norm.weight.data)
        torch.nn.init.zeros_(self.decoder_norm.bias.data)

    def forward(self, pixel_values, gsd, output_attentions: bool = False, mask = None):
        h, _ = self.patch_embedding(pixel_values)
        h = self.enc_positional_embedding(h, gsd)
        h, mask, ids_restore = self.random_masking(h, mask=mask)

        # add CLS token (has no positional encoding)
        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # encoder
        h, _ = self.encoder(h, output_attentions=output_attentions)

        # latent
        h = self.enc_dec_projection(h)
        h = self.random_masking.add_mask_tokens_and_unshuffle(h, ids_restore)
        h = self.dec_positional_embedding(h, gsd)

        # decoder
        h, _ = self.decoder(h, output_attentions=output_attentions)
        h = self.decoder_norm(h)
        h = self.decoder_pred(h)

        # remove CLS
        h = h[:, 1:, :]

        output = self.unpatchify(h)
        return {
            'logits': output,
            'loss': self.loss(pixel_values, output),
            'mask': mask,
        }

    def unpatchify(self, patchified_pixel_values):
        """
        (in)  patchified_pixel_values - (batch_size, num_patches, patch_size**2 * num_channels)
        (out) pixel_values - (batch_size, num_channels, height, width)
        """
        patch_size = self.config.patch_size
        num_patches_y = self.config.image_size[0] // patch_size
        num_patches_x = self.config.image_size[1] // patch_size
        num_channels = self.config.num_channels

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_y,
            num_patches_x,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_y * patch_size,
            num_patches_x * patch_size,
        )
        return pixel_values
    
    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean()
