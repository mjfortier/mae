import os
import torch
import pandas as pd
import numpy as np
import json
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)


class FluxDataset(Dataset):
    def __init__(
            self, data_dir, sites, time_series=False, context_length=48,
            target_columns=['NEE_VUT_REF'],
            remove_columns=['timestamp', 'NEE_VUT_REF', 'GPP_NT_VUT_REF', 'RECO_NT_VUT_REF'], # to be removed by default from predictors
            ):
        self.data_dir = data_dir
        self.sites = sites
        self.data = []
        self.time_series = time_series
        self.context_length = context_length

        self.target_columns = target_columns
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

    def num_channels(self):
        # returns number of frequency bands in the imagery
        _, _, modis = self.data[0]
        return modis[list(modis.keys())[0]].shape[0]
    
    def columns(self):
        _, labels, _, _, _ = self.__getitem__(0)
        return labels
    
    def mask_targets(self, prev_targets):
        # Add targets to predictors, but only a random number of them to simulate cold starts
        prev_mask = torch.zeros(prev_targets.shape).to(torch.bool)
        n = np.random.randint(0, len(prev_targets))
        prev_mask[-1:] = True
        prev_mask[:n] = True
        return prev_mask | prev_targets.isnan()

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
        
        predictor_df = rows.drop(columns=self.remove_columns)
        labels = list(predictor_df.columns)
        target_df = rows[self.target_columns]
        
        predictors = torch.tensor(predictor_df.values)
        mask = predictors.isnan()
        predictors = predictors.nan_to_num(-1.0) # just needs a numeric value, doesn't matter what

        targets = torch.tensor(target_df.values[-1:])

        if self.time_series:
            prev_targets = torch.tensor(target_df.values)
            prev_mask = self.mask_targets(prev_targets)

            predictors = torch.cat([predictors, prev_targets], dim=1)
            mask = torch.cat([mask, prev_mask], dim=1)
            labels.extend(self.target_columns)

        return predictors, labels, mask, modis_data, targets


def custom_collate_fn(batch):
    predictors, labels, mask, modis_data, targets = zip(*batch)
    # Normal attributes
    predictors = torch.stack(predictors, dim=0)
    mask = torch.stack(mask, dim=0)
    targets = torch.stack(targets, dim=0)

    for l in labels[1:]:
        np.testing.assert_array_equal(labels[0], l, f'Difference found in input arrays {labels[0]} and {l}')
    labels = labels[0]

    # List of modis data. Tuples of (batch, timestep, data)
    modis_list = []
    for b, batch in enumerate(modis_data):
        for t, data in batch:
            modis_list.append((b, t, data))
    modis_data = modis_list

    return predictors, labels, mask, modis_data, targets


class FluxTimeSeriesValidationDataLoader():
    def __init__(
            self, data_dir, sites, context_length=48,
            batch_size=32,
            target_columns=['NEE_VUT_REF'],
            time_columns=['DOY', 'TOD'],
            remove_columns=['timestamp', 'NEE_VUT_REF', 'GPP_NT_VUT_REF', 'RECO_NT_VUT_REF'],
            ):
        self.data_dir = data_dir
        self.sites = sites
        self.context_length = context_length
        self.batch_size = batch_size
        self.data = []

        self.target_columns = target_columns.copy()
        self.time_columns = time_columns.copy()
        self.remove_columns = remove_columns.copy()
        #[self.remove_columns.remove(c) for c in self.target_columns]

        self.current_files = []

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
                cache_df = df[['timestamp'] + target_columns].copy()
                df[float_cols] = df[float_cols].astype(np.float32)

                with open(os.path.join(root, 'modis.pkl'), 'rb') as f:
                    modis_data = pkl.load(f)
                with open(os.path.join(root, 'meta.json'), 'r') as f:
                    meta = json.load(f)

                self.data.append({
                    'meta': meta,
                    'df': df,
                    'cache_df': cache_df,
                    'modis_data': modis_data,
                    'final_row': len(df),
                    'current_row': self.context_length - 1,
                    'finished': False
                })
        self.set_dataset_len()
    
    def reset_files(self):
        for file in self.data:
            file['cache_df'][self.target_columns] = np.nan
            file['cache_df'][self.target_columns] = file['cache_df'][self.target_columns].astype(np.float32)
            file['current_row'] = self.context_length
            file['finished'] = False
        self.set_current_files()

    def set_dataset_len(self):
        self.reset_files()
        i = 0
        while len(self.current_files) > 0:
            dirty_current_file_list = False
            for file in self.current_files:
                file['current_row'] += 1
                if file['current_row'] > len(file['df']):
                    file['finished'] = True
                    dirty_current_file_list = True
            if dirty_current_file_list:
                self.set_current_files()
            i += 1
        self.len = i
        self.reset_files()

    def __len__(self):
        return self.len
    
    def set_current_files(self):
        # Set the current_files array with up to {context_length} unfinished files
        for file in self.data:
            if file['finished']:
                if file in self.current_files:
                    self.current_files.remove(file)
                continue
            if file not in self.current_files and len(self.current_files) < self.batch_size:
                self.current_files.append(file)
    
    def update_inferred_values(self, output_values):
        dirty_current_file_list = False
        if type(output_values) != list:
            output_values = [output_values]
        for file, target_values in zip(self.current_files, output_values):
            file['cache_df'].loc[file['current_row']-1, self.target_columns] = target_values
            file['current_row'] += 1
            if file['current_row'] > len(file['df']):
                file['finished'] = True
                dirty_current_file_list = True
        if dirty_current_file_list:
            self.set_current_files()
    
    def get_batch(self):
        ### This is what needs to be finished
        batch = []
        for file in self.current_files:
            row_max = file['current_row']
            row_min = row_max - self.context_length
            rows = file['df'].iloc[row_min:row_max]
            rows = rows.reset_index(drop=True)
            cache_rows = file['cache_df'].iloc[row_min:row_max]
            cache_rows = cache_rows.reset_index(drop=True)

            modis_data = []
            timestamps = list(rows['timestamp'])
            for i, ts in enumerate(timestamps):
                pixels = file['modis_data'].get(ts, None)
                if pixels is not None:
                    modis_data.append((i, torch.tensor(pixels[:,1:9,1:9], dtype=torch.float32)))
        


            predictor_df = rows.drop(columns=self.remove_columns)
            labels = list(predictor_df.columns)
            target_df = rows[self.target_columns]
            
            predictors = torch.tensor(predictor_df.values)
            mask = predictors.isnan()
            predictors = predictors.nan_to_num(-1.0) # just needs a numeric value, doesn't matter what

            targets = torch.tensor(target_df.values[-1:])

            cache_df = cache_rows.drop(columns=['timestamp'])
            cache_values = torch.tensor(cache_df.values)
            cache_mask = cache_values.isnan()
            cache_values = cache_values.nan_to_num(-1.0)

            predictors = torch.cat([predictors, cache_values], dim=1)
            mask = torch.cat([mask, cache_mask], dim=1)
            labels.extend(list(cache_df.columns))

            batch.append([predictors, labels, mask, modis_data, targets])
        return custom_collate_fn(batch) if len(batch) > 0 else []

    def batches(self):
        self.reset_files()
        batch = self.get_batch()
        while len(batch) > 0:
            yield batch
            batch = self.get_batch()

    def __iter__(self):
        return self.batches()
    
    def inference_values(self):
        files = []
        for file in self.data:
            cache_df = file['cache_df'][['NEE_VUT_REF']]
            df = pd.concat([file['df'][['timestamp', 'NEE_VUT_REF']], cache_df.rename(columns={'NEE_VUT_REF': 'inferred'})], axis=1)
            files.append([file['meta'], df])
        return files
        

def FluxDataLoader(data_dir, sites, context_length=48, target_columns=['NEE_VUT_REF'], time_series=False, **kwargs):
    ds = FluxDataset(data_dir, sites, context_length=context_length, target_columns=target_columns, time_series=time_series)
    return DataLoader(ds, collate_fn=custom_collate_fn, **kwargs)
