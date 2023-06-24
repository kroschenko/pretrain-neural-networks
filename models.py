import torch
import torch.nn as nn
from common_types import InitTypes


class UnifiedClassifier(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.a_functions = []
        print(layers_config["architecture"])
        self.layers_config = layers_config["architecture"]
        layer_index = 0
        for layer in self.layers_config:
            if len(layer[0]) == 2:
                new_layer = nn.Linear(*layer[0])
            if len(layer[0]) == 3:
                new_layer = nn.Conv2d(*layer[0])
            self.layers.append(new_layer)
            self.a_functions.append(layer[1][-1])
            layer_index += 1

    def forward(self, x):
        for layer, afunc, layer_config in zip(self.layers, self.a_functions, self.layers_config):
            x = afunc(layer(x))
            if len(layer_config) == 3:
                post_processing_actions = layer_config[2]
                for action in post_processing_actions:
                    x = action(x)
        return x


class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, a_func, init_type, without_sampling):
        super(RBM, self).__init__()
        if init_type == InitTypes.Kaiming:
            W = torch.empty(n_vis, n_hid)
            v = torch.empty(1, n_vis)
            h = torch.empty(1, n_hid)
            torch.nn.init.kaiming_normal_(W, mode='fan_out', nonlinearity="relu")
            torch.nn.init.kaiming_normal_(v, mode='fan_out', nonlinearity="relu")
            torch.nn.init.kaiming_normal_(h, mode='fan_out', nonlinearity="relu")
            self.W = nn.Parameter(W)
            self.v = nn.Parameter(v)
            self.h = nn.Parameter(h)
        elif init_type == InitTypes.SimpleNormal:
            self.W = nn.Parameter(0.01 * torch.randn(n_vis, n_hid))
            self.v = nn.Parameter(0.01 * torch.randn(1, n_vis))
            self.h = nn.Parameter(0.01 * torch.randn(1, n_hid))
        elif init_type == InitTypes.SimpleUniform:
            self.W = nn.Parameter(0.02 * torch.rand(n_vis, n_hid)-0.01)
            self.v = nn.Parameter(0.02 * torch.rand(1, n_vis)-0.01)
            self.h = nn.Parameter(0.02 * torch.rand(1, n_hid)-0.01)
        self.a_func = a_func
        self.without_sampling = without_sampling

    def visible_to_hidden(self, v):
        weighted_sum = torch.mm(v, self.W) + self.h
        output = self.a_func[1](weighted_sum)
        return output, weighted_sum

    def hidden_to_visible(self, h):
        weighted_sum = torch.mm(h, self.W.t()) + self.v
        output = self.a_func[0](weighted_sum)
        return output, weighted_sum

    def forward(self, v0):
        h0, h0_ws = self.visible_to_hidden(v0)
        if self.without_sampling:
            h_sampled = h0
        else:
            if self.a_func[1] == torch.sigmoid:
                h_sampled = 1. * (h0 > torch.rand(h0.shape))
            elif self.a_func[1] == torch.relu:
                h0_std = torch.std(h0, dim=0, unbiased=False)
                h_sampled = h0 + torch.normal(0, h0_std)
            else:
                h_sampled = h0
        v1, v1_ws = self.hidden_to_visible(h_sampled)
        h1, h1_ws = self.visible_to_hidden(v1)
        return v0, v1, v1_ws, h0, h0_ws, h1, h1_ws

    def __str__(self) -> str:
        return "vis: " + str(self.v.shape) + \
               "hid: " + str(self.h.shape)


class CRBM(nn.Module):
    def __init__(self, n_vis_channels, n_hid_channels, kernel_size, a_func, init_type, without_sampling):
        super(CRBM, self).__init__()
        if init_type == InitTypes.Kaiming:
            W = torch.empty(n_hid_channels, n_vis_channels, kernel_size, kernel_size)
            torch.nn.init.kaiming_normal_(W, nonlinearity="relu")
            self.W = nn.Parameter(W)
        elif init_type == InitTypes.SimpleNormal:
            self.W = nn.Parameter(0.01 * torch.randn(n_hid_channels, n_vis_channels, kernel_size, kernel_size))
            self.v = nn.Parameter(0.01 * torch.randn(1, n_vis_channels, 1, 1))
            self.h = nn.Parameter(0.01 * torch.randn(1, n_hid_channels, 1, 1))
        elif init_type == InitTypes.SimpleUniform:
            self.W = nn.Parameter(0.02 * torch.rand(n_hid_channels, n_vis_channels, kernel_size, kernel_size) - 0.01)
            self.v = nn.Parameter(0.02 * torch.rand(1, n_vis_channels, 1, 1) - 0.01)
            self.h = nn.Parameter(0.02 * torch.rand(1, n_hid_channels, 1, 1) - 0.01)
        self.a_func = a_func
        self.without_sampling = without_sampling

    def visible_to_hidden(self, v):
        weighted_sum = torch.convolution(v, self.W, None, stride=[1,1], padding=[0,0], dilation=[1,1], transposed=False, output_padding=[0,0], groups=1)
        output = self.a_func[1](weighted_sum+self.h)
        return output, weighted_sum

    def hidden_to_visible(self, h):
        weighted_sum = torch.conv_transpose2d(h, self.W, None, stride=[1, 1], padding=[0, 0], output_padding=[0, 0], groups=1,
                               dilation=[1, 1])
        output = self.a_func[0](weighted_sum+self.v)
        return output, weighted_sum

    def forward(self, v0):
        h0, h0_ws = self.visible_to_hidden(v0)
        if self.without_sampling:
            h_sampled = h0
        else:
            if self.a_func[1] == torch.sigmoid:
                h_sampled = 1. * (h0 > torch.rand(h0.shape))
            elif self.a_func[1] == torch.relu:
                h0_std = torch.std(h0, dim=0, unbiased=False)
                h_sampled = h0 + torch.normal(0, h0_std)
            else:
                h_sampled = h0
        v1, v1_ws = self.hidden_to_visible(h_sampled)
        h1, h1_ws = self.visible_to_hidden(v1)
        return v0, v1, v1_ws, h0, h0_ws, h1, h1_ws

    def __str__(self) -> str:
        return "vis: " + str(self.v.shape) + \
               "hid: " + str(self.h.shape)
