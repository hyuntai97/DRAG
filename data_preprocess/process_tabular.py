import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.data[idx]), (self.labels[idx]), torch.tensor([0])

def load_data(path):
    train_data = np.load(os.path.join(path, 'train_data.npy'), allow_pickle = True)
    train_lab = np.ones((train_data.shape[0])) #All positive labelled data points collected
    test_data = np.load(os.path.join(path, 'test_data.npy'), allow_pickle = True)
    test_lab = np.load(os.path.join(path, 'test_labels.npy'), allow_pickle = True)

    ## preprocessing 
    mean=np.mean(train_data,0)
    std=np.std(train_data,0)
    train_data=(train_data-mean)/ (std + 1e-4)
    num_features = train_data.shape[1]
    test_data = (test_data - mean)/(std + 1e-4)

    train_samples = train_data.shape[0]
    test_samples = test_data.shape[0]
    print("Train Samples: ", train_samples)
    print("Test Samples: ", test_samples)

    uniq_cnt = np.unique(test_lab, return_counts=True)[1]
    ratio = uniq_cnt[0] * 100 / (uniq_cnt[0] + uniq_cnt[1]) 
    print('Anomaly_ratio: ', ratio)
    
    return CustomDataset(train_data, train_lab), CustomDataset(test_data, test_lab), num_features, ratio
