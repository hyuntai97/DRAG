'''
Code borrowed from https://github.com/lukasruff/Deep-SVDD-PyTorch 
'''
from PIL import Image
import numpy as np
from random import sample 
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Subset
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__

class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader

class FMNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)
        #Loads only the digit 0 and digit 1 data
        # for both train and test 
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        mean = [
            [0.3256056010723114],
            [0.22290456295013428],
            [0.376699835062027],
            [0.25889596343040466],
            [0.3853232264518738],
            [0.1367349475622177],
            [0.3317836821079254],
            [0.16769391298294067],
            [0.35355499386787415],
            [0.30119451880455017]
        ]
        std = [
            [0.35073918104171753],
            [0.34353047609329224],
            [0.3586803078651428],
            [0.3542196452617645],
            [0.37631189823150635],
            [0.26310813426971436],
            [0.3392786681652069],
            [0.29478660225868225],
            [0.3652712106704712],
            [0.37053292989730835]
        ]
        # if normal_classes in [0, 1]
        if normal_class in [3, 5, 6, 7, 8, 9]:   
            print('1')                                    
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean[normal_class], std[normal_class]) ]) 
            tr_transform = transforms.Compose([
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(180),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean[normal_class], std[normal_class]) ]) 
        else:
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5],
                                            std=[0.5])])
            tr_transform = transforms.Compose([
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(180),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5],
                                            std=[0.5])])
                                             
        target_transform = transforms.Lambda(lambda x: int(x not in self.outlier_classes))

        train_set = MyFMNIST(root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)
        self.test_set = MyFMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)

class MyFMNIST(FashionMNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyFMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
 

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x
