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
        [self.remove_columns.remove(c) for c in self.target_columns]

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
                gt_df = df[['timestamp'] + target_columns].copy()
                df[float_cols] = df[float_cols].astype(np.float32)
                df[self.target_columns] = np.nan
                df[self.target_columns] = df[self.target_columns].astype(np.float32)

                with open(os.path.join(root, 'modis.pkl'), 'rb') as f:
                    modis_data = pkl.load(f)
                with open(os.path.join(root, 'meta.json'), 'r') as f:
                    meta = json.load(f)

                self.data.append({
                    'meta': meta,
                    'df': df,
                    'gt_df': gt_df,
                    'modis_data': modis_data,
                    'final_row': len(df),
                    'current_row': self.context_length - 1,
                    'finished': False
                })
        self.set_dataset_len()
    
    def reset_files(self):
        for file in self.data:
            file['df'][self.target_columns] = np.nan
            file['df'][self.target_columns] = file['df'][self.target_columns].astype(np.float32)
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
            file['df'].loc[file['current_row']-1, self.target_columns] = target_values
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

            modis_data = []
            timestamps = list(rows['timestamp'])
            for i, ts in enumerate(timestamps):
                pixels = file['modis_data'].get(ts, None)
                if pixels is not None:
                    modis_data.append((i, torch.tensor(pixels[:,1:9,1:9], dtype=torch.float32)))
        
            predictor_df = rows.drop(columns=self.remove_columns + self.time_columns)
            time_df = rows[self.time_columns]
            target_df = rows[self.target_columns]
        
            predictor_values = torch.tensor(predictor_df.values)
            predictor_labels = list(predictor_df.columns) # may be different from self.tabular_columns
            predictor_mask = predictor_values.isnan()
            predictor_values = predictor_values.nan_to_num(-1.0) # just needs a numeric value, doesn't matter what

            time_values = torch.tensor(time_df.values)
            time_labels = list(time_df.columns)

            target_values = torch.tensor(target_df.values[-1:])
            target_labels = list(target_df.columns)
            batch.append([
                predictor_values, predictor_labels, predictor_mask, time_values, time_labels, modis_data, target_values, target_labels
            ])
        return custom_collate_fn(batch) if len(batch) > 0 else []

    def batches(self):
        self.reset_files()
        batch = self.get_batch()
        while len(batch) > 0:
            yield batch
            batch = self.get_batch()

    def __iter__(self):
        return self.batches()
        

def FluxDataLoader(data_dir, sites, context_length=48, target_columns=['NEE_VUT_REF'], time_series=False, **kwargs):
    ds = FluxDataset(data_dir, sites, context_length=context_length, target_columns=target_columns, time_series=time_series)
    return DataLoader(ds, collate_fn=custom_collate_fn, **kwargs)
