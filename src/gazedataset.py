import json
import os, sys
from torch.utils.data import Dataset

class Round2_Dataset(Dataset):
    def __init__(self):
        f = open(os.path.join(os.path.dirname(__file__), "gaze_data/Round2_data.json"))
        self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]