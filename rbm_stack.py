import torch
from torch import nn
import utilities as utl
from config import Config
from models import RBM, CRBM
from common_types import PretrainingType, LayerTrainType
from data_config import get_tensor_dataset_from_loader


class RBMStack:
    def __init__(self, layers_config, device, init_type, without_sampling, k):
        self.rbm_stack = []
        self.layers = layers_config["architecture"]
        self.input_dim = layers_config["input_dim"]
        self.device = device
        for i in range(0, len(self.layers) - 1):
            layer_params = self.layers[i][0]
            layer_activation_function = self.layers[i][1]
            rbm_constructor = RBM if len(layer_params) == 2 else CRBM
            rbm = rbm_constructor(*layer_params, layer_activation_function, init_type, without_sampling, k)
            self.rbm_stack.append(rbm.to(self.device))

    def _prepare_train_set(self, train_set, batch_size, input_dim, layer_index=0):
        batches_count = len(train_set) / batch_size
        if isinstance(input_dim, tuple):
            resulted_array = torch.zeros((train_set.shape[0], *input_dim))
        else:
            resulted_array = torch.zeros((train_set.shape[0], input_dim))
        i = 0
        while i < batches_count:
            _slice = slice(i * batch_size, (i + 1) * batch_size)
            inputs = train_set[_slice]
            if layer_index == 0:
                resulted_array[_slice] = inputs
            else:
                inputs = inputs.to(self.device)
                h0, _ = self.rbm_stack[layer_index - 1].visible_to_hidden(inputs)
                # v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = self.rbm_stack[layer_index - 1](inputs)
                resulted_array[_slice] = h0
            i += 1
        if len(self.layers[layer_index - 1]) == 3:
            post_processing_actions = self.layers[layer_index - 1][2]
            for action in post_processing_actions:
                if not isinstance(action, torch.nn.Dropout):
                    resulted_array = action(resulted_array)
        return resulted_array

    def train(self, train_loader, pretrain_type, layer_train_type):
        layers_losses = {}
        layer_index = 0
        if layer_train_type == LayerTrainType.PerLayer:
            train_set = get_tensor_dataset_from_loader(train_loader)
            train_set = self._prepare_train_set(train_set, Config.pretraining_batch_size, self.input_dim)
            batches_count = len(train_set) / Config.pretraining_batch_size
            with torch.no_grad():
                for rbm in self.rbm_stack:
                    if pretrain_type == PretrainingType.Hybrid:
                        current_pretrain = PretrainingType.RBMClassic if layer_index == 0 else PretrainingType.REBA
                    else:
                        current_pretrain = pretrain_type
                    print(current_pretrain)
                    # layers_losses["layer_"+str(layer_index)], \
                    output_shape = utl.train_rbm_with_custom_dataset(
                        train_set, self.device, rbm, current_pretrain, batches_count)
                    print(output_shape)
                    layer_index += 1
                    train_set = self._prepare_train_set(train_set, Config.pretraining_batch_size, output_shape[1:],
                                                        layer_index)
        if layer_train_type == LayerTrainType.PerBatch:
            with torch.no_grad():
                for epoch in range(Config.pretraining_epochs):
                    loss = 0
                    for i, data in enumerate(train_loader, 0):
                        batch = data[0].to(self.device)
                        # print(batch.shape)
                        for index in range(0, layer_index):
                            batch, _ = self.rbm_stack[index].visible_to_hidden(batch)
                            if len(self.layers[index]) == 3:
                                post_processing_actions = self.layers[index][2]
                                for action in post_processing_actions:
                                    if not isinstance(action, torch.nn.Dropout):
                                        batch = action(batch)
                        # print(batch.shape)
                        rbm = self.rbm_stack[layer_index]
                        if pretrain_type == PretrainingType.Hybrid:
                            current_pretrain = PretrainingType.RBMClassic if layer_index == 0 else PretrainingType.REBA
                        else:
                            current_pretrain = pretrain_type
                        loss += utl.train_rbm_with_batch(rbm, batch, current_pretrain).item()
                        layer_index = (layer_index + 1) % len(self.rbm_stack)
                    print(loss)
                        # print(layer_index)

        return layers_losses

    def torch_model_init_from_weights(self, torch_model):
        with torch.no_grad():
            for i in range(0, len(torch_model.layers) - 1):
                if len(torch_model.layers_config[i][0]) == 2:
                    torch_model.layers[i].weight.data = self.rbm_stack[i].W.T
                    torch_model.layers[i].bias.data = torch.reshape(self.rbm_stack[i].h, (len(self.rbm_stack[i].h[0]),))
                if len(torch_model.layers_config[i][0]) == 3:
                    torch_model.layers[i].weight.data = self.rbm_stack[i].W
                    torch_model.layers[i].bias.data = self.rbm_stack[i].h.reshape(torch_model.layers[i].bias.data.shape)

    def do_reduction(self, layers_config):
        with torch.no_grad():
            for i in range(0, len(self.layers) - 1):
                mask = torch.abs(self.rbm_stack[i].W) > Config.reduction_param
                # reduction_params_count = (~mask).sum() * 100. / mask.numel()
                self.rbm_stack[i].W *= mask.double()
            condition = None
            for i in range(0, len(self.layers) - 1):
                if condition is not None:
                    self.rbm_stack[i].W = nn.Parameter(self.rbm_stack[i].W[condition])
                condition = torch.abs(self.rbm_stack[i].W).sum(dim=0) != 0
                # removed_columns_indices = torch.where(~condition)[0]
                self.rbm_stack[i].W = nn.Parameter(self.rbm_stack[i].W[:, condition])
                self.rbm_stack[i].h = nn.Parameter(self.rbm_stack[i].h[:, condition])
                layers_config["architecture"][i][0] = tuple(self.rbm_stack[i].W.shape)
            previous_neurons_count = layers_config["architecture"][len(self.layers)-2][0][1]
            end_layer_neurons_count = layers_config["architecture"][len(self.layers)-1][0][1]
            layers_config["architecture"][len(self.layers)-1][0] = (previous_neurons_count, end_layer_neurons_count)
