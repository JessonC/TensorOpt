# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class MIMODataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.X = torch.from_numpy(data['X']).cfloat()
        self.H = torch.from_numpy(data['H']).cfloat()
        self.Y = torch.from_numpy(data['Y']).cfloat()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.Y[idx], self.X[idx], self.H[idx]
