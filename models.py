import torch
import torch.nn as nn
from common_types import InitTypes


class UnifiedClassifier(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.a_functions = []
        layers = layers_config["architecture"]
        if len(layers_config["activation"]) == 1:
            a_functions = layers_config["activation"] * (len(layers) - 1)
        else:
            a_functions = layers_config["activation"]
        for i in range(0, len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.a_functions.append(a_functions[i])

    def forward(self, x):
        for i in range(0, len(self.layers)-1):
            x = self.a_functions[i]((self.layers[i](x)))
        x = self.layers[len(self.layers)-1](x)
        return x


class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, a_func, init_type, without_sampling):
        super(RBM, self).__init__()
        if init_type == InitTypes.Kaiming:
            W = 0.01 * torch.randn(n_vis, n_hid)
            torch.nn.init.kaiming_normal_(W, mode='fan_out', nonlinearity="relu")
            self.W = nn.Parameter(W)
            self.v = nn.Parameter(0.01 * torch.randn(1, n_vis))
            self.h = nn.Parameter(0.01 * torch.randn(1, n_hid))
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
        output = self.a_func(weighted_sum)
        return output, weighted_sum

    def hidden_to_visible(self, h):
        weighted_sum = torch.mm(h, self.W.t()) + self.v
        output = self.a_func(weighted_sum)
        return output, weighted_sum

    def forward(self, v0):
        h0, h0_ws = self.visible_to_hidden(v0)
        if self.without_sampling:
            h_sampled = h0
        else:
            if self.a_func == torch.sigmoid:
                h_sampled = 1. * (h0 > torch.rand(h0.shape))
            elif self.a_func == torch.relu:
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
    def __init__(self, n_vis, n_hid, kernel_size, a_func, init_type, without_sampling):
        super(RBM, self).__init__()
        if init_type == InitTypes.Kaiming:
            conv = torch.nn.Conv2d(n_vis, n_hid, kernel_size)
            torch.nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            self.W = nn.Parameter(conv.weight)
            self.v = nn.Parameter(0.01 * torch.randn(1, n_vis))
            self.h = nn.Parameter(0.01 * torch.randn(1, n_hid))
        elif init_type == InitTypes.SimpleNormal:
            conv = torch.nn.Conv2d(n_vis, n_hid, kernel_size)
            self.W = nn.Parameter(0.01 * torch.randn(conv.weight.data))
            self.v = nn.Parameter(0.01 * torch.randn(1, n_vis))
            self.h = nn.Parameter(0.01 * torch.randn(1, n_hid))
    # elif init_type == InitTypes.SimpleUniform:
    #     self.W = nn.Parameter(0.02 * torch.rand(n_vis, n_hid)-0.01)
    #     self.v = nn.Parameter(0.02 * torch.rand(1, n_vis)-0.01)
    #     self.h = nn.Parameter(0.02 * torch.rand(1, n_hid)-0.01)
        # if init_type == InitTypes.Kaiming:
        #     W = 0.01 * torch.randn(n_vis, n_hid)
        #     torch.nn.init.kaiming_normal_(W, mode='fan_out')
        #     self.W = nn.Parameter(W)
        #     self.v = nn.Parameter(0.01 * torch.randn(1, n_vis))
        #     self.h = nn.Parameter(0.01 * torch.randn(1, n_hid))
        # elif init_type == InitTypes.SimpleNormal:
        #     self.W = nn.Parameter(0.01 * torch.randn(n_vis, n_hid))
        #     self.v = nn.Parameter(0.01 * torch.randn(1, n_vis))
        #     self.h = nn.Parameter(0.01 * torch.randn(1, n_hid))
        # elif init_type == InitTypes.SimpleUniform:
        #     self.W = nn.Parameter(0.02 * torch.rand(n_vis, n_hid)-0.01)
        #     self.v = nn.Parameter(0.02 * torch.rand(1, n_vis)-0.01)
        #     self.h = nn.Parameter(0.02 * torch.rand(1, n_hid)-0.01)
        # self.a_func = a_func
        # self.without_sampling = without_sampling

    def visible_to_hidden(self, v):
        weighted_sum = torch.mm(v, self.W) + self.h
        output = self.a_func(weighted_sum)
        return output, weighted_sum

    def hidden_to_visible(self, h):
        weighted_sum = torch.mm(h, self.W.t()) + self.v
        output = self.a_func(weighted_sum)
        return output, weighted_sum

    def forward(self, v0):
        h0, h0_ws = self.visible_to_hidden(v0)
        if self.without_sampling:
            h_sampled = h0
        else:
            if self.a_func == torch.sigmoid:
                h_sampled = 1. * (h0 > torch.rand(h0.shape))
            elif self.a_func == torch.relu:
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
