import os
import torch
from typing import Tuple
from torch import nn
import vit_foundry.vit_components as vc
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import pandas as pd
from einops import rearrange
import numpy as np
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
    weight_sharing: bool = False
    mlp_ratio: int = 3
    num_frequencies: int = 12
    context_length: int = 64
    num_heads: int = 8
    obs_dropout: float = 0.0
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
    def __init__(
            self, data_dir, sites, time_series=False, context_length=48,
            target_columns=['NEE_VUT_REF'],
            time_columns=['DOY', 'TOD'],
            remove_columns=['timestamp', 'NEE_VUT_REF', 'GPP_NT_VUT_REF', 'RECO_NT_VUT_REF'], # to be removed by default from predictors
            ):
        self.data_dir = data_dir
        self.sites = sites
        self.data = []
        self.time_series = time_series
        self.context_length = context_length

        self.target_columns = target_columns
        self.time_columns = time_columns
        self.remove_columns = remove_columns
        
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
        
        self.tabular_columns = list(col_df.columns) + self.target_columns # total list for model config purposes
        self.modis_bands = max([v.shape[0] for v in list(self.data[0][2].values())])

    def num_channels(self):
        # returns number of frequency bands in the imagery
        _, _, modis = self.data[0]
        return modis[list(modis.keys())[0]].shape[0]
    
    def add_masked_targets(self, predictor_df, target_df):
        # Add targets to predictors, but only a random number of them to simulate cold starts
        n = np.random.randint(0, len(predictor_df))
        target_df[-1:] = np.nan
        target_df[:n] = np.nan
        predictor_df = pd.concat([predictor_df, target_df], axis=1)
        return predictor_df

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
        
        predictor_df = rows.drop(columns=self.remove_columns + self.time_columns)
        time_df = rows[self.time_columns]
        target_df = rows[self.target_columns]
        if self.time_series:
            predictor_df = self.add_masked_targets(predictor_df, target_df.copy())
        
        predictor_values = torch.tensor(predictor_df.values)
        predictor_labels = list(predictor_df.columns) # may be different from self.tabular_columns
        predictor_mask = predictor_values.isnan()
        predictor_values = predictor_values.nan_to_num(-1.0) # just needs a numeric value, doesn't matter what

        time_values = torch.tensor(time_df.values)
        time_labels = list(time_df.columns)

        target_values = torch.tensor(target_df.values[-1:])
        target_labels = list(target_df.columns)

        return predictor_values, predictor_labels, predictor_mask, time_values, time_labels, modis_data, target_values, target_labels


def custom_collate_fn(batch):
    predictor_values, predictor_labels, predictor_mask, time_values, time_labels, modis_data, target_values, target_labels = zip(*batch)

    # Normal attributes
    predictor_values = torch.stack(predictor_values, dim=0)
    predictor_mask = torch.stack(predictor_mask, dim=0)
    time_values = torch.stack(time_values, dim=0)
    target_values = torch.stack(target_values, dim=0)

    # List of modis data. Tuples of (batch, timestep, data)
    modis_list = []
    for b, batch in enumerate(modis_data):
        for t, data in batch:
            modis_list.append((b, t, data))
    
    # Ensure all samples have the same label number and order
    for a in predictor_labels[1:]:
        np.testing.assert_array_equal(predictor_labels[0], a, f'Difference found in input arrays {predictor_labels[0]} and {a}')

    return predictor_values, predictor_labels[0], predictor_mask, time_values, time_labels[0], modis_list, target_values, target_labels[0]

def FluxDataLoader(data_dir, sites, context_length=48, target_columns=['NEE_VUT_REF'], time_series=False, **kwargs):
    ds = FluxDataset(data_dir, sites, context_length=context_length, target_columns=target_columns, time_series=time_series)
    return DataLoader(ds, collate_fn=custom_collate_fn, **kwargs)


class Perceiver(nn.Module):
    def __init__(self, config: PerceiverConfig):
        super().__init__()
        self.config = config
        self.input_embeddings = nn.Embedding(len(self.config.tabular_inputs), self.config.input_embedding_dim)
        self.input_embeddings_map = {label: i for i, label in enumerate(self.config.tabular_inputs)}

        self.fourier = FourierFeatureMapping(self.config.num_frequencies)
        self.input_hidden_dim = self.config.input_embedding_dim + self.config.num_frequencies * 2
        self.obs_dropout = nn.Dropout(p=self.config.obs_dropout)

        latent_hidden_dim = self.config.latent_hidden_dim
        context_length = self.config.context_length

        self.latent_embeddings = nn.Embedding(context_length, latent_hidden_dim)

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
        if self.config.weight_sharing:
            cross_attention_block = [
                vc.AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio, kv_hidden_size=self.input_hidden_dim),
                vc.AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio)
            ]
            for i in range(len(self.layer_types)//2):
                block_type = self.layer_types[i*2:(i+1)*2]
                if block_type == 'cs':
                    layers.extend(cross_attention_block)
                else:
                    layers.extend([
                        vc.AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio),
                        vc.AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio)
                    ])
            
        else:
            for l in self.layer_types:
                if l == 'c':
                    layers.append(
                        vc.AttentionLayer(latent_hidden_dim, config.num_heads, config.mlp_ratio, kv_hidden_size=self.input_hidden_dim)
                    )
                elif l == 's':
                    layers.append(
                        vc.AttentionLayer(config.latent_hidden_dim, config.num_heads, config.mlp_ratio)
                    )

        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(latent_hidden_dim, 1)
        self.causal_mask = nn.Parameter(torch.zeros((1, context_length, context_length), dtype=torch.bool), requires_grad=False)
        if self.config.causal:
            for y in range(context_length):
                for x in range(context_length):
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

    #def forward(self, observations, masks, spectral_data, fluxes):
    def forward(
            self, predictor_values, predictor_labels, predictor_mask,
            time_values, time_labels, spectral_data,
            fluxes, flux_labels):
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

        # Marshall data
        predictor_mask = predictor_mask.to(device)
        time_mask = torch.zeros(time_values.shape, device=device).to(torch.bool)
        if self.training:
            dropout_mask = ~self.obs_dropout(torch.ones(predictor_mask.shape, device=device)).to(torch.bool)
            predictor_mask = predictor_mask | dropout_mask
        
        masks = torch.cat([time_mask, predictor_mask], dim=-1) # so time values are never masked out
        predictor_values = predictor_values.to(device)
        time_values = time_values.to(device)
        observations = torch.cat([time_values, predictor_values], dim=-1)
        labels = time_labels + predictor_labels
        fluxes = fluxes.to(device)
        if len(spectral_data) == 0:
            return self.forward_no_images(observations, labels, masks, fluxes, flux_labels)

        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = torch.stack([self.input_embeddings.weight[self.input_embeddings_map[l]] for l in labels], dim=0).to(device)
        embedding_obs = embedding_obs.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
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
        
        op = self.output_proj(hidden[:,-1,:]).squeeze() # B
        loss = self.loss(fluxes.squeeze(), op)
        
        return {
            'loss': loss,
            'logits': op,
        }
    
    def forward_no_images(self, observations, labels, masks, fluxes, flux_labels):
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
        B, L, P = observations.shape
        fourier_obs = self.fourier(observations) # (B, L, P, 2*F)
        embedding_obs = torch.stack([self.input_embeddings.weight[self.input_embeddings_map[l]] for l in labels], dim=0).to(device)
        embedding_obs = embedding_obs.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1) # (B, L, P, I)
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
        
        op = self.output_proj(hidden[:,-1,:]).squeeze() # B
        loss = self.loss(fluxes.squeeze(), op)
        return {
            'loss': loss,
            'logits': op,
        }
    
    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean()
