import torch
import torchvision.transforms as transforms
import utilities as utl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from common_types import DatasetType


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
    #  transforms.Lambda(_flatten)
     ]
)


transform_CIFAR = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(_flatten)])


def get_torch_dataset(dataset_type, batch_size):
    dataset_selector = {
        DatasetType.MNIST: transform_MNIST,
        DatasetType.CIFAR10: transform_CIFAR,
    }
    if dataset_type == DatasetType.IRIS:
        train_set = IrisDataset("./data/fisher_irises/iris_train.txt")  # 120 items
        test_set = IrisDataset("./data/fisher_irises/iris_test.txt")  # 30
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    else:
        transform = dataset_selector[dataset_type]
        dataset_con = utl.get_dataset_constructor(dataset_type)
        train_set = dataset_con(root='./data', train=True, download=True, transform=transform)
        test_set = dataset_con(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_set, test_set, train_loader, test_loader


def get_tensor_dataset_from_loader(torch_loader):
    dataset_lst = []
    i = 0
    for data, label in torch_loader:
        dataset_lst.append(data)
        i += 1
    return torch.vstack(dataset_lst)
