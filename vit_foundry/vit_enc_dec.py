import os
import torch
from torch import nn
import vit_foundry.vit_components as vc
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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
class EncDecDataset(Dataset):
    def __init__(self, root_dir, split="train", in_memory=False):
        """
        Args:
            root_dirs (list[string]): One or more root dir paths. Each should have a 'train' and 'test' subdir
            split (string): Either "train" or "test"
        """
        self.root_dir = root_dir
        self.split = split
        self.in_memory = in_memory
        self.transform = transforms.ToTensor()
        self.image_files = []
        
        if self.in_memory:
            for file in os.listdir(os.path.join(self.root_dir, split, 'mask')):
                pre_rgb_file = os.path.join(self.root_dir, self.split, 'pre_rgb', file)
                post_rgb_file = os.path.join(self.root_dir, self.split, 'post_rgb', file)
                mask_file = os.path.join(self.root_dir, self.split, 'mask', file)

                pre_rgb = self.transform(Image.open(pre_rgb_file))
                post_rgb = self.transform(Image.open(post_rgb_file))
                mask = self.transform(Image.open(mask_file)).squeeze()

                self.image_files.append((pre_rgb, post_rgb, mask))
        else:
            for file in os.listdir(os.path.join(self.root_dir, split, 'mask')):
                self.image_files.append(file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.image_files[idx]
        else:
            file = self.image_files[idx]
            pre_rgb_file = os.path.join(self.root_dir, self.split, 'pre_rgb', file)
            post_rgb_file = os.path.join(self.root_dir, self.split, 'post_rgb', file)
            mask_file = os.path.join(self.root_dir, self.split, 'mask', file)

            pre_rgb = self.transform(Image.open(pre_rgb_file))
            post_rgb = self.transform(Image.open(post_rgb_file))
            mask = self.transform(Image.open(mask_file)).squeeze()

            return pre_rgb, post_rgb, mask


class EncDecViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = vc.ViTMAEPatchEmbeddings(config)
        self.positional_embedding = vc.ViTMAEPositionalEmbeddings(config.image_size, config.patch_size, config.hidden_size)
        self.random_masking = vc.ViTMAERandomMasking(config)
        self.enc_dec = vc.ViTEncoderDecoder(config)
        self.pixel_projection = nn.Linear(config.hidden_size, config.patch_size * config.patch_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights()

    def initialize_weights(self):
        #torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
        torch.nn.init.normal_(self.pixel_projection.weight.data, std=self.config.initializer_range)
        #torch.nn.init.normal_(self.decoder_pred.weight.data, std=self.config.initializer_range)
        #torch.nn.init.ones_(self.decoder_norm.weight.data)
        #torch.nn.init.zeros_(self.decoder_norm.bias.data)
        
    def masked_forward(self, enc_pixel_values, dec_pixel_values, output_attentions=False, mask=None):
        enc_h = self.patch_embedding(enc_pixel_values)
        dec_h = self.patch_embedding(dec_pixel_values)
        enc_h = self.positional_embedding(enc_h)
        dec_h = self.positional_embedding(dec_h)
        
        return enc_h, dec_h 

    def forward(self, enc_pixel_values, dec_pixel_values, output_attentions: bool = False, mask = None):
        enc_h = self.patch_embedding(enc_pixel_values)
        dec_h = self.patch_embedding(dec_pixel_values)
        enc_h = self.positional_embedding(enc_h)
        dec_h = self.positional_embedding(dec_h)


        enc_h, dec_h, _ = self.enc_dec(enc_h, dec_h, output_attentions=output_attentions)
        # We only use the decoder tokens for final image
        h = self.pixel_projection(dec_h)

        output = self.unpatchify(h)
        return self.sigmoid(output)

    def unpatchify(self, patchified_pixel_values):
        """
        (in)  patchified_pixel_values - (batch_size, num_patches, patch_size**2 * num_channels)
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
