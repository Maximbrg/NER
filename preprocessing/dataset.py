import pandas as pd
from torch.utils.data import Dataset


class CustomNERDataset(Dataset):

    def __init__(self, file_path: str = None):
        self.df = pd.read_csv(file_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['A'][idx], self.df['C'][idx]