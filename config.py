import torch.nn as nn
from torch.nn.modules.module import Module, Tensor
from common_types import DatasetType, InitTypes, LayerTrainType, PretrainingType
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
    pretraining_epochs = 10
    pretraining_rate = 0.000125  # 0.00002 # 0.001   0.00001 - MNIST
    pretraining_rate_reba = 0.000125   # 0.00002 # 0.001  0.00004 - MNIST

    finetune_rate = 0.001
    max_finetuning_epochs = 15
    finetuning_momentum = 0.9
    test_every_epochs = 1
    count_attempts_in_experiment = 1
    init_type = InitTypes.SimpleNormal
    with_sampling = False
    # with_reduction = False
    with_adaptive_rate = False
    # reduction_param = 0.001
    reduction_after_epochs = 10
    layer_train_type = LayerTrainType.PerLayer
    use_validation_dataset = False
    validation_split_value = 0.9
    validate_every_epochs = 1
    validation_decay = 3
    test_batch_size = 128
    freeze_pretrained_layers = False
    pretraining_schemes = [
        [PretrainingType.RBMClassic, True], [PretrainingType.Without, False], [PretrainingType.Without, True]
    ]
    DATASETS = [DatasetType.CIFAR100]


relu = nn.ReLU()
rrelu = nn.RReLU()
linear = Linear()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)
logsoftmax = nn.LogSoftmax(dim=1)
add_postprocessing = nn.Flatten()
unflatten = nn.Unflatten(1, (20, 12, 12))
dropout = nn.Dropout(p=0.5)
dropout_conv = nn.Dropout2d(p=0)
pooling = nn.MaxPool2d(kernel_size=2)
bn32 = nn.BatchNorm2d(32, affine=False).cuda(0)
bn64 = nn.BatchNorm2d(64, affine=False).cuda(0)
bn128 = nn.BatchNorm2d(128, affine=False).cuda(0)
bn_fc = nn.BatchNorm1d(512, affine=False).cuda(0)


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
                [(1, 16, 5, False), [sigmoid, relu], [pooling]],
                [(16, 8, 5, False), [relu, relu], [pooling, add_postprocessing]],
                [(128, 64), [relu, relu]],
                [(64, 32), [relu, relu]],
                [(32, 10), [logsoftmax]],
            ], "input_dim": (1, 28, 28)},
            # {"architecture": [
            #     [(1, 32, 3, True), [relu, tanh]],
            #     [(32, 32, 3, True), [tanh, relu], [pooling, dropout_conv]],
            #     [(32, 64, 3, True), [relu, tanh]],
            #     [(64, 64, 3, True), [tanh, relu], [pooling, dropout_conv]],
            #     [(64, 128, 3, False), [relu, tanh]],
            #     [(128, 128, 3, False), [tanh, relu], [dropout_conv, add_postprocessing]],
            #     [(1152, 512), [relu, tanh], [dropout]],
            #     [(512, 10), [softmax]],
            # ], "input_dim": (1, 28, 28)},
        ],
        DatasetType.CIFAR10: [
            # {"architecture": [
            #     [(3, 64, 5, False), [tanh, relu], [pooling]],
            #     [(64, 32, 5, False), [relu, tanh], [pooling, add_postprocessing]],
            #     [(800, 128), [tanh, relu]],
            #     [(128, 10), [logsoftmax]],
            # ], "input_dim": (3, 32, 32)},
            {"architecture": [
                [(3, 32, 3, True), [tanh, relu]],
                [(32, 32, 3, True), [relu, tanh], [pooling, dropout_conv]],
                [(32, 64, 3, True), [tanh, relu]],
                [(64, 64, 3, True), [relu, tanh], [pooling, dropout_conv]],
                [(64, 128, 3, True), [tanh, relu]],
                [(128, 128, 3, True), [relu, tanh], [pooling, dropout_conv, add_postprocessing]],
                [(2048, 512), [tanh, relu], [bn_fc, dropout]],
                [(512, 10), [logsoftmax]],
            ], "input_dim": (3, 32, 32)},
        ],
        DatasetType.CIFAR100: [
            # {"architecture": [
            #   [(3, 128, 5, False), [sigmoid, relu], [pooling]],
            #   [(128, 64, 5, False), [relu, tanh], [pooling, add_postprocessing]],
            #   [(1600, 1024), [tanh, relu]],
            #   [(1024, 512), [relu, tanh]],
            #   [(512, 100), [logsoftmax]],
            # ], "input_dim": (3, 32, 32)},
            {"architecture": [
                [(3, 32, 3, True), [tanh, relu]],
                [(32, 32, 3, True), [relu, tanh], [pooling, dropout_conv]],
                [(32, 64, 3, True), [tanh, relu]],
                [(64, 64, 3, True), [relu, tanh], [pooling, dropout_conv]],
                [(64, 128, 3, True), [tanh, relu]],
                [(128, 128, 3, True), [relu, tanh], [pooling, dropout_conv, add_postprocessing]],
                [(2048, 512), [tanh, relu], [bn_fc, dropout]],
                [(512, 100), [logsoftmax]],
            ], "input_dim": (3, 32, 32)},
        ]
    }
    return layers_config_selector[experiment_dataset_name]
