import pandas as pd
import numpy as np
import random

import torch

class WeldingDataset():
    def __init__(self,
                 root='data/welding/data.xlsx',
                 mode = 'train',
                 train_num:int = 380,
                 test_num:int = 20,
                 scale:bool = True):
        super().__init__()
        
        if train_num + test_num != 400:
            raise Exception('Check number of samples')

        self.root = root
        self.data_li = []
        
        self.scale = scale
        
        self._check()
        self._preprocess()
        
        random.seed(42)
        random.shuffle(self.data_li)
        
        if mode == 'train':
            self.data_li = self.data_li[:train_num]
        elif mode == 'test':
            self.data_li = self.data_li[train_num:]
        else:
            raise Exception('Check mode')
        
    def __len__(self):
        return len(self.data_li)

    def __getitem__(self, idx):
        sample, label = self.data_li[idx]
        
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return sample, label
        
    def _check(self):
        df = pd.read_excel(self.root, sheet_name=None)
        
        sheet_names = list(df.keys())
        self.data_li = []
        
        for sheet_name in sheet_names:
            if sheet_name not in ['TIG-알루미늄(BOP)', 'TIG-연강(BOP']:
                pass
            idx_li = [i for i in range(3, 28)]
            idx_li.extend([i for i in range(37, 62)])
            for i in idx_li:
                sample = [sheet_name] 
                sample.extend(list(df[sheet_name].iloc[i]))
                self.data_li.append(sample)
    
    def _preprocess(self):
        # 1) Sheet Name
        sheet_dict = {
            'TIG-알루미늄(BOP)': 0,
            'TIG-알루미늄(T-Fillet)': 1,
            'TIG-알루미늄(V-Groove)': 2,
            'TIG-알루미늄(I-Groove)': 3,
            'TIG-연강(BOP': 4,
            'TIG-연강(T-Fillet)': 5,
            'TIG-연강(V-Groove)': 6,
            'TIG-연강(I-Groove)': 7
        }
        for data in self.data_li:
            data[0] = sheet_dict[data[0]]
        
        # 2) Drop NaN, and Estimate ID, Estimate Identification
        for data in self.data_li:
            del data[1:4]
        
        # 3) Convert to NumPy
        self.data_li = np.array(self.data_li)
        
        # 4) Split Label
        labels = self.data_li[:, 8:26]
        self.data_li = self.data_li[:, :8]
        
        # 5) Min-Max Scaling according to each column axis
        if self.scale == True:
            for i in range(1, 8):
                X_min = self.data_li[:, i].min()
                X_max = self.data_li[:, i].max()
        
                scaled_col = (self.data_li[:, i] - X_min) / (X_max - X_min)
                self.data_li[:, i] = scaled_col
        
        # 6) Label ((x+y),) -> (2, (x|y))
        seq_labels = []
        for label in labels:
            label_seq = []
            for x, y in zip(label[0::2], label[1::2]):
                label_seq.append([x, y])
            seq_labels.append(label_seq)

        # 7) Merge Sample and Label
        self.data_li = self.data_li.tolist()
        
        for i in range(0, len(self.data_li)):
            self.data_li[i] = (np.array(self.data_li[i]), np.array(seq_labels[i]))
        
        # 8) Use Embedding in nn.Embedding
        # So, don't use encoding method like one-hot encoding here
        
if __name__ == '__main__':
    ds = WeldingDataset()
    data, label = ds[251]
    print(data)
    print(label)
    print(label.shape)
    
    import matplotlib.pyplot as plt
    
    plt.plot(label[:, 0], label[:, 1])
    plt.savefig('test.png')