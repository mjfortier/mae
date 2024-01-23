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


class MultiImageDataset(Dataset):
    def __init__(self, root_dirs, split="train"):
        """
        Args:
            root_dirs (list[string]): One or more root dir paths. Each should have a 'train' and 'test' subdir
            split (string): Either "train" or "test"
        """
        self.root_dirs = root_dirs
        self.split = split
        self.transform = transforms.ToTensor()
        self.image_files = []
        
        for i, root_dir in enumerate(root_dirs):
            self.image_files.append([f for f in os.listdir(os.path.join(root_dir, split)) if f.endswith('.png')])
            self.image_files[i].sort()

    def __len__(self):
        return len(self.image_files[0])

    def __getitem__(self, idx):
        images = []
        for i, root_dir in enumerate(self.root_dirs):
            images.append(
                self.transform(
                    Image.open(os.path.join(root_dir, self.split, self.image_files[i][idx]))
                )
            )

        return torch.cat(images, axis=1)


class ViTMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = vc.ViTMAEPatchEmbeddings(config)
        self.enc_positional_embedding = vc.ViTSinCosPositionalEmbeddings(config.image_size, config.patch_size, config.hidden_size)
        self.random_masking = vc.ViTMAERandomMasking(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.encoder = vc.ViTMAEEncoder(config)
        self.enc_dec_projection = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.dec_positional_embedding = vc.ViTSinCosPositionalEmbeddings(config.image_size, config.patch_size, config.decoder_hidden_size)
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

    def forward(self, pixel_values, output_attentions: bool = False, mask = None):
        h = self.patch_embedding(pixel_values)
        h = self.enc_positional_embedding(h)
        h, mask, ids_restore = self.random_masking(h, mask=mask)

        # add CLS token (has no positional encoding)
        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # encoder
        h, _ = self.encoder(h, output_attentions=output_attentions)

        # latent
        h = self.enc_dec_projection(h)
        h = self.random_masking.add_mask_tokens_and_unshuffle(h, ids_restore)
        h = self.dec_positional_embedding(h)

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
            'aerial_loss': self.loss(pixel_values[:,:256,:], output[:,:256,:]), # This is hacked in; change it soon.
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
    

    ## Multi-task mask generation

    def fill_in_middle_mask(self):
        num_patches_x = self.config.image_size[1] // self.config.patch_size
        num_patches_y = self.config.image_size[0] // self.config.patch_size

        mask = torch.zeros(num_patches_y, num_patches_x)

        if num_patches_x == num_patches_y:
            # square image
            mask[2:num_patches_y - 2, 2:num_patches_x - 2] = 1
        else:
            # rectangular image
            mask[2:(num_patches_y // 2) - 2, 2:num_patches_x - 2] = 1
        return mask.flatten().unsqueeze(0)


    def random_task_mask(self):
        if torch.rand(1) > 1:
            return self.fill_in_middle_mask()
        else:
            return None
