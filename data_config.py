import config
import torch
import torchvision.transforms as transforms
import utilities as utl


def _flatten(x):
    return torch.flatten(x)


transform_MNIST = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(_flatten)])


transform_CIFAR = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(_flatten)])


def get_torch_dataset(dataset_type, batch_size):
    dataset_selector = {
        utl.DatasetType.MNIST: transform_MNIST,
        utl.DatasetType.CIFAR10: transform_CIFAR,
    }
    transform = dataset_selector[dataset_type]
    dataset_con = utl.get_dataset_constructor(dataset_type)
    train_set = dataset_con(root='./data', train=True, download=True, transform=transform)
    test_set = dataset_con(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_set, test_set, train_loader, test_loader
