from torch.utils.data import Dataset
import numpy as np

class DemoDataset(Dataset):
    def __init__(self):
        self.x = np.random.randn(15).reshape((5,3))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]
    