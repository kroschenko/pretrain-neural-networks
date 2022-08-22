import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Cifar10Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
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
        # output = torch.sigmoid(F.linear(v, self.W, self.h))
        return output

    def hidden_to_visible(self, h):
        output = torch.sigmoid(torch.mm(h, self.W.t()) + self.v)
        # output = torch.sigmoid(F.linear(h, self.W.t(), self.v))
        return output

    # def free_energy(self, v):
    #     r"""Free energy function.
    #     .. math::
    #         \begin{align}
    #             F(x) &= -\log \sum_h \exp (-E(x, h)) \\
    #             &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
    #         \end{align}
    #     Args:
    #         v (Tensor): The visible variable.
    #     Returns:
    #         FloatTensor: The free energy value.
    #     """
    #     v_term = torch.matmul(v, self.v.t())
    #     w_x_h = F.linear(v, self.W, self.h)
    #     h_term = torch.sum(F.softplus(w_x_h), dim=1)
    #     return torch.mean(-h_term - v_term)

    def forward(self, v0):
        # v0 = torch.flatten(v0, 1)
        h0 = self.visible_to_hidden(v0)
        h_sampled = h0#1. * (h0 > torch.rand(h0.shape).to('cuda:0'))
        v1 = self.hidden_to_visible(h_sampled)
        h1 = self.visible_to_hidden(v1)
        return v0, v1, h0, h1

    def __str__(self) -> str:
        return "vis: " + str(self.v.shape) + \
               "hid: " + str(self.h.shape)
