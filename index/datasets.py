import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm

class EmbDataset(data.Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        names = ['emb']
        usecols = [1]
        tsv_data = pd.read_csv(data_path, sep = '\t',usecols = usecols, names = names, quotechar = None, quoting = 3)
        features = tsv_data['emb'].values.tolist()
        num_data = len(features)
        for i in tqdm(range(num_data)):
            features[i] = [float(s) for s in features[i].split(' ')]
        self.embeddings = np.array(features, dtype = np.float16)
        assert self.embeddings.shape[0] == num_data
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.tensor(emb, dtype = torch.float16)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
