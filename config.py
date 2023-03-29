import torch.nn as nn
from common_types import DatasetType, InitTypes
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    project_name: str = "kroschenko/pretrain-networks"
    api_token: str = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NDM2YWM0Yy1iMGIxLTQwZTctYjMwNy04YWFiY2QxZjgzOWEifQ=="

max_random_seed = 1024

pretraining_batch_size = 64
momentum_beg = 0
momentum_end = 0
momentum_change_epoch = 5
pretraining_epochs = 20
pretraining_rate = 0.0005 # 0.001
pretraining_rate_reba = 0.0005 # 0.001

finetune_rate = 0.01
finetuning_epochs = 60
finetuning_momentum = 0.9
test_every_epochs = 10
count_attempts_in_experiment = 1
init_type = InitTypes.SimpleNormal
without_sampling = True

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)
add_postprocessing = nn.Flatten()
unflatten = nn.Unflatten(1, (20, 5, 5))
dropout = nn.Dropout(p=0.5)
pooling = nn.MaxPool2d(kernel_size=2)

def get_layers_config_for_dataset(experiment_dataset_name):
    layers_config_selector = {
        # utl.DatasetType.IRIS: [
        #     {"architecture": [4, 10, 10, 3], "activation": [torch.relu]}
        # ]
        DatasetType.MNIST: [
            # {"architecture": [
            #     [(784, 800), a_func],
            #     [(800, 800), a_func],
            #     [(800, 10), last_layer_a_func]
            # ], "input_dim": 784},
            # {"architecture": [
            #     [(1, 20, 5), relu, [pooling]],
            #     [(20, 40, 5), relu, [pooling, add_postprocessing]],
            #     [(640, 1000), relu, [dropout]],
            #     [(1000, 1000), relu, [dropout]],
            #     [(1000, 10), softmax],
            # ], "input_dim": (1, 28, 28)},
            {"architecture": [
                [(1, 20, 5), relu, [pooling, add_postprocessing]],
                [(2880, 2880), relu, [dropout, unflatten]],
                [(20, 40, 5), relu, [pooling, add_postprocessing]],
                [(640, 1000), relu, [dropout]],
                [(1000, 1000), relu, [dropout]],
                [(1000, 10), softmax],
            ], "input_dim": (1, 28, 28)},
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
