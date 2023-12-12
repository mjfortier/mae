import os
import torch
from torch import nn
import vit_foundry.vit_components as vc
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def vit_mae_base_config(**kwargs):
    return vc.ViTMAEConfig(
        hidden_size = 768,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        decoder_hidden_size = 512,
        decoder_num_hidden_layers = 8,
        decoder_num_attention_heads = 16,
        **kwargs
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


class MaskClassifierDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.transform = transforms.ToTensor()
        self.image_files = [f for f in os.listdir(os.path.join(root_dir, split, 'rgb'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        rgb = self.transform(Image.open(os.path.join(self.root_dir, self.split, 'rgb', self.image_files[idx])))
        mask = self.transform(Image.open(os.path.join(self.root_dir, self.split, 'mask', self.image_files[idx])))
        return rgb, mask


class StackedMaskClassifierDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.transform = transforms.ToTensor()
        self.image_files = [f for f in os.listdir(os.path.join(root_dir, split, 'mask'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        pre_rgb = self.transform(Image.open(os.path.join(self.root_dir, self.split, 'pre_rgb', self.image_files[idx])))
        post_rgb = self.transform(Image.open(os.path.join(self.root_dir, self.split, 'post_rgb', self.image_files[idx])))
        rgb = torch.vstack((pre_rgb, post_rgb))
        mask = self.transform(Image.open(os.path.join(self.root_dir, self.split, 'mask', self.image_files[idx])))
        return rgb, mask


class ViTMaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = vc.ViTMAEPatchEmbeddings(config)
        self.enc_positional_embedding = vc.ViTMAEPositionalEmbeddings(config.image_size, config.patch_size, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.encoder = vc.ViTMAEEncoder(config)
        self.mask_pred = nn.Linear(
            config.hidden_size, config.patch_size * config.patch_size, bias=True
        )
        self.activation = nn.Sigmoid()
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
        torch.nn.init.normal_(self.mask_pred.weight.data, std=self.config.initializer_range)

    def forward(self, pixel_values, output_attentions: bool = False):
        '''
        pixel_values - (B, C, H, W)
        mask - (B, H, W)
        '''
        h = self.patch_embedding(pixel_values)
        h = self.enc_positional_embedding(h)

        # add CLS token (has no positional encoding)
        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # encoder
        h, _ = self.encoder(h, output_attentions=output_attentions)

        # remove CLS
        h = h[:, 1:, :]
        h = self.mask_pred(h)
        h = self.activation(h)

        output = self.unpatchify(h)
        return output

    def unpatchify(self, patchified_pixel_values):
        """
        (in)  patchified_pixel_values - (batch_size, num_patches, patch_size**2)
        (out) pixel_values - (batch_size, num_channels, height, width)
        """
        patch_size = self.config.patch_size
        num_patches_y = self.config.image_size[0] // patch_size
        num_patches_x = self.config.image_size[1] // patch_size

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_y,
            num_patches_x,
            patch_size,
            patch_size,
        )
        patchified_pixel_values = torch.einsum("nhwpq->nhpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_y * patch_size,
            num_patches_x * patch_size,
        )
        return pixel_values
