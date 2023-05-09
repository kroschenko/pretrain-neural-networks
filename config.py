import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F
from torch.nn.modules.module import Module, Tensor
from common_types import DatasetType, InitTypes
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

    def forward(self, input: Tensor) -> Tensor:
        return self.constant * input

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

max_random_seed = 1024
pretraining_batch_size = 64
momentum_beg = 0.5
momentum_end = 0.5
momentum_change_epoch = 5
pretraining_epochs = 10
pretraining_rate = 0.0001#0.00002 # 0.001
pretraining_rate_reba = 0.0001#0.00002 # 0.001

finetune_rate = 0.00005
finetuning_epochs = 50
finetuning_momentum = 0.9
test_every_epochs = 1
count_attempts_in_experiment = 1
init_type = InitTypes.SimpleNormal
without_sampling = True
with_reduction = False
with_adaptive_rate = False
reduction_param = 0.01

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
        # utl.DatasetType.IRIS: [
        #     {"architecture": [4, 10, 10, 3], "activation": [torch.relu]}
        # ]
        DatasetType.MNIST: [
            # {"architecture": [
            #     [(784, 800), sigmoid],
            #     [(800, 800), sigmoid],
            #     [(800, 10), logsoftmax]
            # ], "input_dim": 784},
            # {"architecture": [
            #     [(784, 1600), relu],
            #     [(1600, 1600), relu],
            #     [(1600, 800), relu],
            #     [(800, 800), relu],
            #     [(800, 10), logsoftmax]
            # ], "input_dim": 784},
            {"architecture": [
                [(1, 20, 5), [sigmoid, relu], [pooling]],
                [(20, 40, 5), [relu, tanh], [pooling, add_postprocessing]],
                [(640, 1000), [tanh, relu], [dropout]],
                [(1000, 1000), [relu, tanh], [dropout]],
                [(1000, 10), [logsoftmax]],
            ], "input_dim": (1, 28, 28)},
            # {"architecture": [
            #     [(1, 20, 5), relu, [pooling]],
            #     [(20, 40, 5), relu, [pooling, add_postprocessing]],
            #     [(640, 1000), relu, [dropout]],
            #     [(1000, 1000), relu, [dropout]],
            #     [(1000, 10), softmax],
            # ], "input_dim": (1, 28, 28)},
            # {"architecture": [
            #     [(1, 20, 11), relu, [add_postprocessing]],
            #     [(6480, 2880), relu, [dropout, unflatten]],
            #     [(20, 40, 11), relu, [add_postprocessing]],
            #     [(160, 100), relu, [dropout]],
            #     [(100, 100), relu, [dropout]],
            #     [(100, 10), softmax],
            # ], "input_dim": (1, 28, 28)},
            # {"architecture": [784, 800, 800, 10], "activation": [torch.relu]},
            # {"architecture": [[1, (28, 28)], [6, (11, 11), (18, 18)], [6, (11, 11), (8, 8)], [6, (8, 8), (1, 1)]], "activation": [torch.sigmoid]}
            # {"architecture": [784, 1600, 1600, 800, 800, 10], "activation": [torch.relu]},
            # {"architecture": [[(7, 7), 32], [(5, 5), 16], [(5, 5), 8], 256, 10], "activation": [torch.relu]},
        ],
        # utl.DatasetType.CIFAR10: [
        #     [3072, 1024, 512, 256, 128, 64, 10],
        #     [3072, 512, 256, 128, 64, 10]
        # ],
        # utl.DatasetType.CIFAR100: [
        #     [3072, 1536, 1024, 512, 256, 100]
        # ]
    }
    return layers_config_selector[experiment_dataset_name]
