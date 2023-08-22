import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config
from common_types import DatasetType
import torchvision.datasets as datasets


class IrisDataset(Dataset):
    def __init__(self, csv_file, target_column="species"):
        self.mappers = {}
        self.data = pd.read_csv(csv_file)
        head_names = list(self.data)
        head_names.remove(target_column)
        self.target_frame = self.data[target_column]
        self.samples_frame = self.data[head_names]
        self.samples = self.samples_frame.to_numpy().astype("float32")
        self.target = self.target_frame.to_numpy().astype("long")
        self.samples = (self.samples - self.samples.min(axis=0)) / (self.samples.max(axis=0) - self.samples.min(axis=0))
        print(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.target[idx]
        return sample, target  # as Tuple


def _flatten(x):
    return torch.flatten(x)


transform_MNIST = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Lambda(_flatten)
     ]
)


transform_CIFAR = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.RandomRotation(5),
     # transforms.Lambda(_flatten)
     ]
)

transform_CIFAR_train = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.RandomRotation(5),
        # transforms.RandomAffine(25),
        # transforms.RandomHorizontalFlip(0.5),
    ]
)

transform_COMMON = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def get_dataset_constructor(dataset_type: DatasetType):
    dataset_selector = {
        DatasetType.MNIST: datasets.MNIST,
        DatasetType.CIFAR10: datasets.CIFAR10,
        DatasetType.CIFAR100: datasets.CIFAR100
    }
    return dataset_selector[dataset_type]


def get_data_loaders(dataset_type, batch_size):
    loaders = {}
    dataset_selector = {
        DatasetType.MNIST: transform_MNIST,
        DatasetType.CIFAR10: transform_CIFAR,
        DatasetType.CIFAR100: transform_CIFAR_train,
    }
    if dataset_type == DatasetType.IRIS:
        train_set = IrisDataset("./data/fisher_irises/iris_train.txt")
        test_set = IrisDataset("./data/fisher_irises/iris_test.txt")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_loader = None
    else:
        transform = dataset_selector[dataset_type]
        dataset_con = get_dataset_constructor(dataset_type)
        train_set = dataset_con(root='./data', train=True, download=True, transform=transform)
        test_set = dataset_con(root='./data', train=False, download=True, transform=transform_COMMON)
        val_set = None
        if config.Config.use_validation_dataset:
            train_size = int(len(train_set.data) * config.Config.validation_split_value)
            val_size = len(train_set.data) - train_size
            train_set, val_set = torch.utils.data.random_split(
                train_set, [int(len(train_set.data) * config.Config.validation_split_value), val_size])
            val_set.dataset.transform = transform_COMMON
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False) if config.Config.use_validation_dataset else None
    loaders["train_loader"] = train_loader
    loaders["test_loader"] = test_loader
    loaders["val_loader"] = val_loader
    return loaders


def get_tensor_dataset_from_loader(torch_loader):
    dataset_lst = []
    i = 0
    for data, label in torch_loader:
        dataset_lst.append(data)
        i += 1
    return torch.vstack(dataset_lst)
