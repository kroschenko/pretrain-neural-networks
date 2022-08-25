import torch
import utilities as utl
import config
from models import RBM
import numpy


class RBMStack:
    def __init__(self, layers, device):
        self.rbm_stack = []
        self.layers = layers
        self.device = device
        for i in range(0, len(self.layers) - 1):
            rbm = RBM(self.layers[i], self.layers[i + 1])
            self.rbm_stack.append(rbm.to(self.device))

    def _form_dataset_for_next_layer_with_loader(self, train_set, train_loader, rbm):
        new_train_set = torch.zeros((train_set.data.shape[0], self.layers[1]))
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            v0, v1, h0, h1 = rbm(inputs)
            new_train_set[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size] = h0
        return new_train_set

    def _form_dataset_for_next_layer_with_custom_data(self, train_set, layer, batches_count, rbm):
        new_train_set = torch.zeros((train_set.shape[0], layer))
        i = 0
        while i < batches_count:
            _slice = slice(i * config.pretraining_batch_size, (i + 1) * config.pretraining_batch_size)
            inputs = train_set[_slice].to(self.device)
            v0, v1, h0, h1 = rbm(inputs)
            new_train_set[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size] = h0
            i += 1
        return new_train_set

    def train(self, train_set, train_loader, pretrain_type):
        layers_losses = {}
        with torch.no_grad():
            rbm = self.rbm_stack[0]
            layers_losses["layer_0"] = utl.train_rbm_with_loader(train_loader, self.device, rbm, pretrain_type)
            new_train_set = self._form_dataset_for_next_layer_with_loader(train_set, train_loader, rbm)
            batches_count = len(new_train_set) / config.pretraining_batch_size
            j = 1
            for rbm, layer in zip(self.rbm_stack[1:-1], self.layers[2:-1]):
                layers_losses["layer_"+str(j)] = utl.train_rbm_with_custom_dataset(
                    new_train_set, self.device, rbm, pretrain_type, batches_count
                )
                new_train_set = self._form_dataset_for_next_layer_with_custom_data(new_train_set, layer, batches_count, rbm)
                j += 1
        return layers_losses

    def torch_model_init_from_weights(self, torch_model):
        with torch.no_grad():
            for i in range(0, len(torch_model.layers) - 1):
                torch_model.layers[i].weight.data = self.rbm_stack[i].W.T
                torch_model.layers[i].bias.data = torch.reshape(self.rbm_stack[i].h, (len(self.rbm_stack[i].h[0]),))
