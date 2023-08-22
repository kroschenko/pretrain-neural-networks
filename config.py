import torch.nn as nn
from torch.nn.modules.module import Module, Tensor
from common_types import DatasetType, InitTypes, LayerTrainType
from dataclasses import dataclass


@dataclass
class ProjectConfig:
    project_name: str = "kroschenko/pretrain-networks"
    api_token: str = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NDM2YWM0Yy1iMGIxLTQwZTctYjMwNy04YWFiY2QxZjgzOWEifQ=="


class Linear(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False, constant=1):
        super(Linear, self).__init__()
        self.inplace = inplace
        self.constant = constant

    def forward(self, _input: Tensor) -> Tensor:
        return self.constant * _input

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


@dataclass
class Config:
    max_random_seed = 1024
    pretraining_batch_size = 128
    momentum_beg = 0.5
    momentum_end = 0.9
    momentum_change_epoch = 5
    pretraining_epochs = 1
    pretraining_rate = 0.000125  # 0.00002 # 0.001   0.00001 - MNIST
    pretraining_rate_reba = 0.000125  # 0.00002 # 0.001  0.00004 - MNIST

    finetune_rate = 0.001
    max_finetuning_epochs = 25
    finetuning_momentum = 0.9
    test_every_epochs = 1
    count_attempts_in_experiment = 1
    init_type = InitTypes.SimpleNormal
    with_sampling = False
    with_reduction = False
    with_adaptive_rate = False
    reduction_param = 0.01
    layer_train_type = LayerTrainType.PerLayer
    use_validation_dataset = True
    validation_split_value = 0.9
    validate_every_epochs = 1
    validation_decay = 3
    test_batch_size = 128


relu = nn.ReLU()
linear = Linear()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)
logsoftmax = nn.LogSoftmax(dim=1)
add_postprocessing = nn.Flatten()
unflatten = nn.Unflatten(1, (20, 12, 12))
dropout = nn.Dropout(p=0.2)
pooling = nn.MaxPool2d(kernel_size=2)


def get_layers_config_for_dataset(experiment_dataset_name):
    layers_config_selector = {
        DatasetType.MNIST: [
            # {"architecture": [
            #     [(784, 1600), relu],
            #     [(1600, 1600), relu],
            #     [(1600, 800), relu],
            #     [(800, 800), relu],
            #     [(800, 10), logsoftmax]
            # ], "input_dim": 784},
            {"architecture": [
                [(1, 40, 5), [sigmoid, relu], [pooling]],
                [(40, 40, 5), [relu, relu], [pooling, add_postprocessing]],
                [(640, 320), [relu, relu]],
                [(320, 160), [relu, relu]],
                [(160, 10), [logsoftmax]],
            ], "input_dim": (1, 28, 28)},
        ],
        DatasetType.CIFAR10: [
            {"architecture": [
                [(3, 64, 5), [sigmoid, relu], [pooling]],
                [(64, 32, 5), [relu, tanh], [pooling, add_postprocessing]],
                [(800, 128), [tanh, relu]],
                [(128, 10), [logsoftmax]],
            ], "input_dim": (3, 32, 32)},
        ],
        DatasetType.CIFAR100: [
            {"architecture": [
              [(3, 128, 5), [sigmoid, relu], [pooling]],
              [(128, 64, 5), [relu, tanh], [pooling, add_postprocessing]],
              [(1600, 1024), [tanh, relu]],
              [(1024, 512), [relu, tanh]],
              [(512, 100), [logsoftmax]],
            ], "input_dim": (3, 32, 32)},
        ]
    }
    return layers_config_selector[experiment_dataset_name]
