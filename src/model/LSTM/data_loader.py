import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FloodTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, seq_len=12, train=True, split_ratio=0.8):
        df = pd.read_csv(csv_path)
        # encode month & region
        df['Month'] = df['Month'].astype('category').cat.codes
        df['Region'] = LabelEncoder().fit_transform(df['Region'])
        df['Flood'] = df['Flood'].map({'No': 0, 'Yes': 1})

        # scale numeric features
        features = ['Year','Month','Region','Avg LST','Avg NDSI','Avg NDVI','Avg Precipitation']
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

        # group by region and build sequences
        sequences = []
        labels = []
        for region, group in df.groupby('Region'):
            group = group.sort_values(['Year','Month'])
            data = group[features].values
            target = group['Flood'].values
            # sliding window
            for i in range(len(group) - seq_len):
                seq = data[i:i+seq_len]
                lbl = target[i+seq_len]   # next month flood label
                sequences.append(seq)
                labels.append(lbl)

        sequences = np.stack(sequences)    # (N, seq_len, n_features)
        labels    = np.array(labels)       # (N,)

        # split
        N = len(labels)
        split = int(N*split_ratio)
        if train:
            self.X = torch.tensor(sequences[:split], dtype=torch.float32)
            self.y = torch.tensor(labels[:split], dtype=torch.float32).unsqueeze(1)
        else:
            self.X = torch.tensor(sequences[split:], dtype=torch.float32)
            self.y = torch.tensor(labels[split:], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(csv_path="data/final/combined-data.csv", seq_len=12, batch_size=32):
    train_ds = FloodTimeSeriesDataset(csv_path, seq_len=seq_len, train=True)
    test_ds  = FloodTimeSeriesDataset(csv_path, seq_len=seq_len, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
