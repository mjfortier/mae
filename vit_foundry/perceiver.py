import os
import torch
from typing import List, Tuple
from torch import nn
import vit_foundry.vit_components as vc
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import pandas as pd
from einops import rearrange
import numpy as np
from time import time
import json
import pickle as pkl
torch.manual_seed(0)


@dataclass
class PerceiverConfig():
    latent_hidden_dim: int = 256
    input_embedding_dim: int = 128
    tabular_inputs: Tuple = ()
    spectral_data_channels: int = 7
    spectral_data_resolution: Tuple = (8,8)
    mlp_ratio: int = 3
    num_frequencies: int = 12
    context_length: int = 64
    num_heads: int = 8
    auxilliary_loss: int = 0
    layers: str = 'cscscsss' # c = cross-attention (with input), s = self-attention
    targets: Tuple = ('GPP_NT_VUT_REF')
    causal: bool = True


class FourierFeatureMapping(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, values):
        embeddings = torch.arange(self.num_frequencies).to(values.device)
        embeddings = torch.pi * 2 ** embeddings
        embeddings = embeddings * values.unsqueeze(-1)
        sin_emb = torch.sin(embeddings)
        cos_emb = torch.cos(embeddings)
        return torch.cat([sin_emb, cos_emb], dim=-1)


class FluxDataset(Dataset):
    def __init__(self, data_dir, sites, context_length=48, targets=['GPP_NT_VUT_REF']):
        self.data_dir = data_dir
        self.sites = sites
        self.data = []
        self.context_length = context_length
        self.targets = targets
        self.remove_columns = ['timestamp', 'NEE_VUT_REF', 'GPP_NT_VUT_REF', 'RECO_NT_VUT_REF']
        
        for root, _, files in os.walk(self.data_dir):
            in_sites = False
            for site in sites:
                if site in root:
                    in_sites = True
            if not in_sites:
                continue
            

            if 'data.csv' in files:
                df = pd.read_csv(os.path.join(root, 'data.csv'))
                float_cols = [c for c in df.columns if c != 'timestamp']
                df[float_cols] = df[float_cols].astype(np.float32)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                with open(os.path.join(root, 'modis.pkl'), 'rb') as f:
                    modis_data = pkl.load(f)
                with open(os.path.join(root, 'meta.json'), 'r') as f:
                    meta = json.load(f)

                self.data.append((meta, df, modis_data))
        
        self.lookup_table = []
        for i, d in enumerate(self.data):
            _, df, _ = d
            for r in range(self.context_length, len(df)+1):
                self.lookup_table.append((i,r))
        
        col_df = self.data[0][1].drop(columns=self.remove_columns)
        self.tabular_columns = list(col_df.columns)
        self.modis_bands = max([v.shape[0] for v in list(self.data[0][2].values())])

    def num_channels(self):
        _, _, modis = self.data[0]
        return modis[list(modis.keys())[0]].shape[0]

    def __len__(self):
        return len(self.lookup_table)

    def __getitem__(self, idx):
        site_num, row_max = self.lookup_table[idx]
        row_min = row_max - (self.context_length)

        _, df, modis = self.data[site_num]
        rows = df.iloc[row_min:row_max]

        rows = rows.reset_index(drop=True)
        modis_data = []
        timestamps = list(rows['timestamp'])
        for i, ts in enumerate(timestamps):
            pixels = modis.get(ts, None)
            if pixels is not None:
                modis_data.append((i, torch.tensor(pixels[:,1:9,1:9], dtype=torch.float32)))
        
        targets = torch.tensor(rows[self.targets].values)
        row_values = torch.tensor(rows.drop(columns=self.remove_columns).values)
        mask = row_values.isnan()
        row_values = row_values.nan_to_num(-1.0) # just needs a numeric value, doesn't matter what
        return row_values, mask, modis_data, targets


def custom_collate_fn(batch):
    row_values, mask, modis_data, targets = zip(*batch)

    # Normal attributes
    row_values = torch.stack(row_values, dim=0)
    mask = torch.stack(mask, dim=0)
    targets = torch.stack(targets, dim=0)

    # List of modis data. Tuples of (batch, timestep, data)
    modis_list = []
    for b, batch in enumerate(modis_data):
        for t, data in batch:
            modis_list.append((b, t, data))

    return row_values, mask, modis_list, targets

def FluxDataLoader(data_dir, sites, context_length = 48, targets=['GPP_NT_VUT_REF'], **kwargs):
    ds = FluxDataset(data_dir, sites, context_length=context_length, targets=targets)
    return DataLoader(ds, collate_fn=custom_collate_fn, **kwargs)


class Perceiver(nn.Module):
    def __init__(self, config: PerceiverConfig):
        super().__init__()
        self.config = config
        self.input_embeddings = nn.Embedding(len(self.config.tabular_inputs), self.config.input_embedding_dim)
        self.fourier = FourierFeatureMapping(self.config.num_frequencies)
        self.input_hidden_dim = self.config.input_embedding_dim + self.config.num_frequencies * 2

        latent_hidden_dim = self.config.latent_hidden_dim
        self.latent_embeddings = nn.Embedding(self.config.context_length, latent_hidden_dim)

        num_pixels = self.config.spectral_data_resolution[0] * self.config.spectral_data_resolution[1]
        self.channels = self.config.spectral_data_channels # for brevity
        self.spectral_projections = nn.ModuleList(
            [nn.Linear(num_pixels, self.config.num_frequencies * 2) for _ in range(self.channels)]
        )
        self.spectral_embeddings = nn.Embedding(self.channels, self.config.input_embedding_dim)
        self.layer_norm_ec = nn.LayerNorm(self.input_hidden_dim, eps=1e-12)
        self.layer_norm_eo = nn.LayerNorm(self.input_hidden_dim, eps=1e-12)

        self.layer_types = self.config.layers
        layers = []
        for l in self.layer_types:
            if l == 'c':
                layers.append(
                    vc.AttentionLayer(config.latent_hidden_dim, config.num_heads, config.mlp_ratio, kv_hidden_size=self.input_hidden_dim)
                )
            elif l == 's':
                layers.append(
                    vc.AttentionLayer(config.latent_hidden_dim, config.num_heads, config.mlp_ratio)
                )

        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(config.latent_hidden_dim, 1)
        self.causal_mask = nn.Parameter(torch.zeros((1, self.config.context_length, self.config.context_length), dtype=torch.bool), requires_grad=False)
        if self.config.causal:
            for y in range(self.config.context_length):
                for x in range(self.config.context_length):
                    self.causal_mask[:,y,x] = y < x
        
        self.apply(self.initialize_weights)
    
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def process_spectral_inputs(self, spectral_data, B, L):
        device = self.input_embeddings.weight.device
        
        imgs = []
        indices = []
        for b, t, img in spectral_data:
            imgs.append(img.flatten(1).to(device))
            indices.append(b*L + t)
        img_data = torch.stack(imgs) # (M, C, num_pixels)
        img_map = torch.zeros(B*L, device=device, dtype=torch.bool)
        img_map[indices] = True
        
        img_data = img_data.transpose(0,1) # (C, M, num_pixels)
        img_data_proj = torch.zeros((self.channels, img_data.shape[1], self.config.num_frequencies * 2), device=device)
        for i, proj in enumerate(self.spectral_projections):
            img_data_proj[i] = proj(img_data[i])
        
        # add embeddings
        img_data = img_data_proj.transpose(0,1) # (M, C, 2*F)
        spec_embeddings = self.spectral_embeddings.weight.unsqueeze(0).repeat(len(spectral_data), 1, 1) # (M, C, I)
        img_data = torch.cat([img_data, spec_embeddings], dim=-1)  # (M, C, IH)
        
        return img_data, img_map

    def forward(self, observations, masks, spectral_data, fluxes):
        '''
        B - batch size
        L - sequence length
        P - # of observations (input variables)
        M - # of observations with images
        F - # of frequencies
        C - # of spectral channels
        I - input embedding dim
        IH - total input dim (I + 2*F)
        H - latent hidden dim
        '''
        if len(spectral_data) == 0:
            return self.forward_no_images(observations, masks, fluxes)
        
        device = self.input_embeddings.weight.device
        observations = observations.to(device)
        masks = masks.to(device)
        fluxes = fluxes.to(device)

        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = self.input_embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
        combined_obs = torch.cat([fourier_obs, embedding_obs], dim=-1) # (B, L, P, IH)
        combined_obs = rearrange(combined_obs, 'B L P IH -> (B L) P IH') # (B*L, P, IH)
        masks = rearrange(masks, 'B L P -> (B L) P').unsqueeze(1) # (B*L, 1, P)

        # images
        img_data, img_map = self.process_spectral_inputs(spectral_data, B, L)

        # divide obs
        masks_with_image = torch.cat([masks[img_map], torch.zeros((len(spectral_data), 1, self.channels), dtype=bool, device=device)], dim=-1) # (M, 1, P+C)
        obs_with_image = torch.cat([combined_obs[img_map], img_data], dim=1)
        masks_without_image = masks[~img_map]
        obs_without_image = combined_obs[~img_map]

        obs_with_image = self.layer_norm_eo(obs_with_image)
        obs_without_image = self.layer_norm_ec(obs_without_image)

        hidden = self.latent_embeddings.weight.unsqueeze(0).repeat(B,1,1) # (B, L, H)

        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'c':
                hidden = rearrange(hidden, 'B L H -> (B L) H').unsqueeze(1) # (B*L, 1, H)
                hidden_with_image = hidden[img_map]
                hidden_without_image = hidden[~img_map]

                hidden_with_image, _ = self.layers[i](hidden_with_image, obs_with_image, mask=masks_with_image)
                hidden_without_image, _ = self.layers[i](hidden_without_image, obs_without_image, mask=masks_without_image)

                hidden[img_map] = hidden_with_image
                hidden[~img_map] = hidden_without_image
                hidden = rearrange(hidden.squeeze(), '(B L) H -> B L H', B=B, L=L)
            elif layer_type == 's':
                hidden, _ = self.layers[i](hidden, hidden, mask=self.causal_mask)
        
        op = self.output_proj(hidden).squeeze()
        all_loss = self.loss(fluxes.squeeze(), op)
        loss = all_loss[-1] + self.config.auxilliary_loss * all_loss[:-1].sum()
        
        return {
            'loss': loss,
            'logits': op,
        }
    
    def forward_no_images(self, observations, masks, fluxes):
        '''
        B - batch size
        L - sequence length
        P - # of observations (input variables)
        M - # of observations with images
        F - # of frequencies
        C - # of spectral channels
        I - input embedding dim
        IH - total input dim (I + 2*F)
        H - latent hidden dim
        '''
        device = self.input_embeddings.weight.device
        observations = observations.to(device)
        masks = masks.to(device)
        fluxes = fluxes.to(device)

        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = self.input_embeddings.weight.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
        combined_obs = torch.cat([fourier_obs, embedding_obs], dim=-1) # (B, L, P, IH)
        combined_obs = rearrange(combined_obs, 'B L P IH -> (B L) P IH') # (B*L, P, IH)
        masks = rearrange(masks, 'B L P -> (B L) P').unsqueeze(1) # (B*L, 1, P)


        combined_obs = self.layer_norm_ec(combined_obs)
        hidden = self.latent_embeddings.weight.unsqueeze(0).repeat(B,1,1) # (B, L, H)

        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'c':
                hidden = rearrange(hidden, 'B L H -> (B L) H').unsqueeze(1) # (B*L, 1, H)

                hidden, _ = self.layers[i](hidden, combined_obs, mask=masks)
                hidden = rearrange(hidden.squeeze(), '(B L) H -> B L H', B=B, L=L)
            elif layer_type == 's':
                hidden, _ = self.layers[i](hidden, hidden, mask=self.causal_mask)
        
        op = self.output_proj(hidden).squeeze()
        all_loss = self.loss(fluxes.squeeze(), op)
        loss = all_loss[-1] + self.config.auxilliary_loss * all_loss[:-1].sum()
        
        return {
            'loss': loss,
            'logits': op,
        }
    
    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean(dim=0)



### Old code

# class PerceiverDataset(Dataset):
#     def __init__(self, met_csv, flux_csv, context_window_length, target='NEE'):
#         self.target = target
#         self.met = pd.read_csv(met_csv, index_col=False).astype(np.float32)
#         self.flux = pd.read_csv(flux_csv, index_col=False).astype(np.float32)
#         self.flux = self.flux[self.flux[target].notna()]
#         self.context = context_window_length

#     def __len__(self):
#         return len(self.flux) - self.context
    
#     def get_columns(self):
#         return list(self.met.drop('year', axis=1).columns)

#     def __getitem__(self, idx):
#         flux_row = self.flux.iloc[idx]
#         year = flux_row['year']
#         doy = flux_row['doy']
#         minutes = flux_row['minutes']
#         row_condition = (self.met['year'] == year) & (self.met['doy'] == doy) & (self.met['minutes'] == minutes)

#         met_row_id = self.met[row_condition].index[0]

#         start, end = max(met_row_id - self.context + 1, 0), met_row_id + 1
#         met_rows = self.met.iloc[start:end]
#         flux_row = flux_row.drop('year')
#         met_rows = met_rows.drop('year', axis=1)
        
#         # Padding for lack of observations
#         met = torch.tensor(met_rows.values)
#         flux = torch.tensor(flux_row[self.target])
        
#         steps, obs = met.shape
#         row_deficit = self.context - steps
#         if row_deficit > 0:
#             padding = torch.empty((row_deficit, obs), dtype=met.dtype)
#             padding[:,:] = float('nan')
#             met = torch.cat((padding, met), 0)

#         return met, flux, met.isnan()





# context_length = 48

# dataset = PerceiverDataset('scotty_creek_met_processed.csv', 'scotty_creek_flux_processed.csv', context_length)
# dl = DataLoader(dataset, batch_size=16, shuffle=True)

# config = PerceiverConfig(
#     inputs=dataset.get_columns(),
#     context_length=context_length,
#     num_frequencies=12,
#     input_embedding_dim=8,
#     latent_hidden_dim=128,
#     num_heads=8,
#     layers = ('cross', 'self', 'cross', 'self', 'cross', 'self', 'self', 'self')
# )
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Perceiver(config)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# def train_one_epoch(model, dataloader, optimizer):
#     total_loss = 0.0
#     for met, flux, mask in tqdm(dataloader):
#         try:
#             met = met.nan_to_num(-9999.0).to(device)
#             flux = flux.to(device)
#             mask = mask.to(device)

#             op = model(met, flux, mask)

#             loss = op['loss']
#             total_loss += loss.item()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         except Exception as e:
#             print(flux)
#     return total_loss / len(dataloader)

# for x in range(5):
#     loss = train_one_epoch(model, dl, optimizer)
#     print(loss)

