from torch.utils.data import Dataset
import numpy as np
import torch


class DemoDataset(Dataset):
    def __init__(self):
        sample_num = 100 * 3
        self.x = np.random.randn(sample_num).reshape((-1,3))
        self.x = self.x.astype(np.float32)
        self.y = np.random.randint(0, 2, sample_num)
        self.y = self.y.astype(np.int64)
        self.y = torch.from_numpy(self.y)
        # print()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    