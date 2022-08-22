import config
import torch
import torchvision.transforms as transforms
import utilities as utl


transform_MNIST = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.flatten(x))])


transform_CIFAR = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.flatten(x))])


def get_torch_dataset(dataset_name, batch_size):
    if dataset_name == utl.DatasetType.MNIST:
        transform = transform_MNIST
    else:
        transform = transform_CIFAR
    dataset_con = utl.get_dataset_constructor(dataset_name)
    trainset = dataset_con(root='./data', train=True,
                                            download=True, transform=transform)
    testset = dataset_con(root='./data', train=False,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainset, testset, trainloader, testloader
