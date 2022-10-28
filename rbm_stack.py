import torch
import utilities as utl
import config
from models import RBM


class RBMStack:
    def __init__(self, layers, a_func, device):
        self.rbm_stack = []
        self.layers = layers
        self.device = device
        for i in range(0, len(self.layers) - 1):
            rbm = RBM(self.layers[i], self.layers[i + 1], a_func)
            self.rbm_stack.append(rbm.to(self.device))

    def _form_dataset_for_next_layer_with_custom_data(self, train_set, layer, batches_count, rbm):
        new_train_set = torch.zeros((train_set.shape[0], layer))
        i = 0
        while i < batches_count:
            _slice = slice(i * config.pretraining_batch_size, (i + 1) * config.pretraining_batch_size)
            inputs = train_set[_slice].to(self.device)
            v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = rbm(inputs)
            new_train_set[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size] = h0
            i += 1
        return new_train_set

    def _prepare_train_set_for_first_rbm(self, train_set, train_loader, batch_size, rbm_inputs_count):
        resulted_array = torch.zeros((train_set.data.shape[0], rbm_inputs_count))
        for i, data in enumerate(train_loader, 0):
            inputs = data[0].to(self.device)
            resulted_array[i * batch_size:(i + 1) * batch_size] = inputs
        return resulted_array

    def train(self, train_set, train_loader, pretrain_type):
        layers_losses = {}
        train_set = self._prepare_train_set_for_first_rbm(
            train_set, train_loader, config.pretraining_batch_size, self.layers[0]
        )
        batches_count = len(train_set) / config.pretraining_batch_size
        with torch.no_grad():
            j = 1
            for rbm, layer in zip(self.rbm_stack[0:-1], self.layers[1:-1]):
                layers_losses["layer_"+str(j)] = utl.train_rbm_with_custom_dataset(
                    train_set, self.device, rbm, pretrain_type, batches_count
                )
                train_set = self._form_dataset_for_next_layer_with_custom_data(train_set, layer, batches_count, rbm)
                j += 1
        return layers_losses

    def torch_model_init_from_weights(self, torch_model):
        with torch.no_grad():
            for i in range(0, len(torch_model.layers) - 1):
                torch_model.layers[i].weight.data = self.rbm_stack[i].W.T
                torch_model.layers[i].bias.data = torch.reshape(self.rbm_stack[i].h, (len(self.rbm_stack[i].h[0]),))
