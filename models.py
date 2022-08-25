import torch
import torch.nn as nn


class UnifiedClassifier(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(0, len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for i in range(0, len(self.layers)-1):
            x = torch.sigmoid(self.layers[i](x))
        x = self.layers[len(self.layers)-1](x)
        return x


class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        # self.W = nn.Parameter(0.02 * torch.rand((n_vis, n_hid)) - 0.01)
        # self.v = nn.Parameter(0.02 * torch.rand((1, n_vis)) - 0.01)
        # self.h = nn.Parameter(0.02 * torch.rand((1, n_hid)) - 0.01)
        self.v = nn.Parameter(torch.zeros(1, n_vis))
        self.h = nn.Parameter(-1 * torch.ones(1, n_hid))
        self.W = nn.Parameter(0.1 * torch.randn(n_vis, n_hid))

    def visible_to_hidden(self, v):
        output = torch.sigmoid(torch.mm(v, self.W) + self.h)
        return output

    def hidden_to_visible(self, h):
        output = torch.sigmoid(torch.mm(h, self.W.t()) + self.v)
        return output

    def forward(self, v0):
        h0 = self.visible_to_hidden(v0)
        h_sampled = h0#1. * (h0 > torch.rand(h0.shape).to('cuda:0'))
        v1 = self.hidden_to_visible(h_sampled)
        h1 = self.visible_to_hidden(v1)
        return v0, v1, h0, h1

    def __str__(self) -> str:
        return "vis: " + str(self.v.shape) + \
               "hid: " + str(self.h.shape)
