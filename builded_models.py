from torch import nn

a_func = nn.ReLU
last_layer_a_func = nn.Softmax

convolutionalModelMNIST = nn.Sequential(
    nn.Conv2d(1, 6, 11),
    a_func(),
    nn.Conv2d(6, 6, 11),
    a_func(),
    nn.Flatten(),
    nn.Linear(384, 120),
    a_func(),
    nn.Linear(120, 10),
    last_layer_a_func()
)

linearNetworkMNIST = nn.Sequential(
    nn.Linear(784, 800),
    a_func(),
    nn.Linear(800, 800),
    a_func(),
    nn.Linear(800, 10),
    last_layer_a_func(),
)
