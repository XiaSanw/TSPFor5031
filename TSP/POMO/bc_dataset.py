import os
import re
import torch
from torch.utils.data import Dataset, DataLoader


class TSPBCDataset(Dataset):
    """Load LKH3-generated (problem, tour) data"""

    def __init__(self, data_path):
        self.data = torch.load(data_path, map_location='cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'problem': torch.from_numpy(item['problem']).float(),
            'tour': torch.from_numpy(item['tour']).long()
        }


def get_bc_dataloader(data_path, batch_size, shuffle=True, num_workers=2):
    """
    Single-scale BC dataloader.

    num_workers recommendations:
    - 2: safe, works on most servers
    - 0: small datasets (<50k/size) or tight memory
    - 4+: very large datasets, but each worker forks a reference to source data
    """
    dataset = TSPBCDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def build_bc_loaders(data_paths, batch_size, num_workers=2):
    """
    Build independent DataLoaders for each problem size.

    Important: different-size samples can't mix in a single batch (tensor shape
    mismatch -> collate error). Training picks a random size per epoch for BC.
    """
    loaders = {}
    for path in data_paths:
        # Infer size from filename, e.g. lkh3_data_n100.pt -> 100
        match = re.search(r'n(\d+)', os.path.basename(path))
        size = int(match.group(1)) if match else 0
        loaders[size] = get_bc_dataloader(path, batch_size, num_workers=num_workers)

    return loaders
