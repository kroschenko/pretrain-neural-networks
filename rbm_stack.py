import random

import torch
from torch import nn
import math
import utilities as utl

import config
from config import Config
from models import RBM, CRBM
from common_types import PretrainingType, LayerTrainType


class RBMStack:
    def __init__(self, layers_config, device, init_type, with_sampling):
        self.rbm_stack = []
        self.layers = layers_config["architecture"]
        self.input_dim = layers_config["input_dim"]
        self.device = device
        for i in range(0, len(self.layers) - 1):
            layer_params = self.layers[i][0]
            layer_activation_function = self.layers[i][1]
            rbm_constructor = RBM if len(layer_params) == 2 else CRBM
            rbm = rbm_constructor(*layer_params, layer_activation_function, init_type, with_sampling, device)
            self.rbm_stack.append(rbm.to(self.device))

    def train(self, loaders, pretrain_type, layer_train_type):
        layers_losses = {}
        layer_index = 0
        if layer_train_type == LayerTrainType.PerLayer:
            with torch.no_grad():
                for rbm in self.rbm_stack:
                    if pretrain_type == PretrainingType.Hybrid:
                        current_pretrain = PretrainingType.RBMClassic if layer_index == 0 else PretrainingType.REBA
                    else:
                        current_pretrain = pretrain_type
                    print(current_pretrain)
                    self.train_rbm_per_layer(loaders, self.device, rbm, current_pretrain, layer_index)
                    layer_index += 1
        if layer_train_type == LayerTrainType.PerBatch:
            with torch.no_grad():
                momentum = Config.momentum_beg
                for epoch in range(Config.pretraining_epochs*len(self.rbm_stack)):
                    loss = 0
                    if epoch == Config.momentum_change_epoch:
                        momentum = Config.momentum_end
                    for i, data in enumerate(loaders["train_loader"]):
                        inputs = data[0].to(self.device)
                        for layer_index in range(0, len(self.rbm_stack)):
                            if pretrain_type == PretrainingType.Hybrid:
                                current_pretrain = PretrainingType.RBMClassic if layer_index == 0 else PretrainingType.REBA
                            else:
                                current_pretrain = pretrain_type
                            rbm = self.rbm_stack[layer_index]
                            train_from_batch_func = self.train_rbm_from_batch if isinstance(rbm, RBM) else self.train_crbm_from_batch
                            vis_loss, hid_loss = train_from_batch_func(rbm, inputs, current_pretrain, momentum)
                            loss += vis_loss.item() + hid_loss.item()
                            inputs, _ = rbm.visible_to_hidden(inputs)
                            if len(self.layers[layer_index]) == 3:
                                post_processing_actions = self.layers[layer_index][2]
                                for action in post_processing_actions:
                                    if not isinstance(action, torch.nn.Dropout) and not isinstance(action,
                                                                                               torch.nn.BatchNorm2d):
                                        inputs = action(inputs)
                        # layer_index = (layer_index + 1) % len(self.rbm_stack)
                    print(loss)
        if layer_train_type == LayerTrainType.PerBatchRandom:
            with torch.no_grad():
                for epoch in range(Config.pretraining_epochs):
                    loss = 0
                    for i, data in enumerate(loaders["train_loader"]):
                        inputs = self.get_data_for_specific_rbm(data[0].to(self.device), layer_index)
                        rbm = self.rbm_stack[layer_index]
                        train_from_batch_func = self.train_rbm_from_batch if isinstance(rbm, RBM) else self.train_crbm_from_batch
                        loss += train_from_batch_func(rbm, inputs, current_pretrain, momentum).item()
                        layer_index = random.randint(0, len(self.rbm_stack)-1)
                    print(loss)
        return layers_losses

    def torch_model_init_from_weights(self, torch_model):
        # with torch.no_grad():
        for i in range(0, len(torch_model.layers) - 1):
            if len(torch_model.layers_config[i][0]) == 2:
                torch_model.layers[i].weight.data = self.rbm_stack[i].weights.T
                torch_model.layers[i].bias.data = torch.reshape(self.rbm_stack[i].h, (len(self.rbm_stack[i].h[0]),))
            if len(torch_model.layers_config[i][0]) >= 3:
                torch_model.layers[i].weight.data = self.rbm_stack[i].weights
                torch_model.layers[i].bias.data = self.rbm_stack[i].h.reshape(torch_model.layers[i].bias.data.shape)
            # torch_model.layers[i].weight.requires_grad = False
            # torch_model.layers[i].bias.requires_grad = False

    @staticmethod
    def train_rbm_from_batch(rbm, batch, pretrain_type, momentum):
        act_func = rbm.a_func
        v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = rbm(batch)
        if pretrain_type == PretrainingType.RBMClassic:
            der_v, der_h = 1, 1
            rate = Config.pretraining_rate
        elif pretrain_type == PretrainingType.REBA:
            der_v = (act_func[0](v1_ws + 0.00001) - act_func[0](v1_ws - 0.00001)) / 0.00002
            der_h = (act_func[1](h1_ws + 0.00001) - act_func[1](h1_ws - 0.00001)) / 0.00002
            rate = Config.pretraining_rate_reba
        if Config.with_adaptive_rate:
            b_h = h0 * ((v1 * v0).sum() + 1) - h1 * (1 + (v1 * v1).sum())
            b_v = v0 * (1 + (h0 * h0).sum()) - v1 * (1 + (h0 * h1).sum())
            rate = (((v0 - v1) * b_v).sum() + ((h0 - h1) * b_h).sum()) / ((b_h * b_h).sum() + (b_v * b_v).sum())
        part_v = (v1 - v0) * der_v
        part_h = (h1 - h0) * der_h
        weights_grad = rate / Config.pretraining_batch_size * (torch.mm(part_v.T, h0) + torch.mm(v1.T, part_h))
        v_thresholds_grad = rate / Config.pretraining_batch_size * part_v.sum(0)
        h_thresholds_grad = rate / Config.pretraining_batch_size * part_h.sum(0)
        rbm.delta_weights = rbm.delta_weights * momentum + weights_grad
        rbm.delta_v_thresholds = rbm.delta_v_thresholds * momentum + v_thresholds_grad
        rbm.delta_h_thresholds = rbm.delta_h_thresholds * momentum + h_thresholds_grad
        rbm.weights -= rbm.delta_weights
        rbm.v -= rbm.delta_v_thresholds
        rbm.h -= rbm.delta_h_thresholds
        vis_part_loss = ((v1 - v0) ** 2).sum()
        hid_part_loss = ((h1 - h0) ** 2).sum()
        return vis_part_loss, hid_part_loss

    @staticmethod
    def train_crbm_from_batch(rbm, batch, pretrain_type, momentum):
        act_func = rbm.a_func
        v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = rbm(batch)
        if pretrain_type == PretrainingType.RBMClassic:
            der_v, der_h = 1, 1
            rate = Config.pretraining_rate
        if pretrain_type == PretrainingType.REBA:
            der_v = (act_func[0](v1_ws + 0.00001) - act_func[0](v1_ws - 0.00001)) / 0.00002
            der_h = (act_func[1](h1_ws + 0.00001) - act_func[1](h1_ws - 0.00001)) / 0.00002
            rate = Config.pretraining_rate_reba
        part_v = (v1 - v0) * der_v
        part_h = (h1 - h0) * der_h
        first_convolution_part = torch.convolution(
            torch.permute(part_v, (1, 0, 2, 3)),
            torch.permute(h0, (1, 0, 2, 3)), None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False,
            output_padding=[0, 0], groups=1)
        second_convolution_part = torch.convolution(
            torch.permute(v1, (1, 0, 2, 3)),
            torch.permute(part_h, (1, 0, 2, 3)), None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False,
            output_padding=[0, 0], groups=1)
        common_conv_expr = first_convolution_part + second_convolution_part

        weights_grad = rate / Config.pretraining_batch_size * torch.permute(common_conv_expr, (1, 0, 2, 3))
        v_thresholds_grad = rate / (Config.pretraining_batch_size * part_v.shape[2] ** 2) * part_v.sum(
            (0, 2, 3)).reshape(
            rbm.v.shape)
        h_thresholds_grad = rate / (Config.pretraining_batch_size * part_h.shape[2] ** 2) * part_h.sum(
            (0, 2, 3)).reshape(
            rbm.h.shape)

        rbm.delta_weights = rbm.delta_weights * momentum + weights_grad
        rbm.delta_v_thresholds = rbm.delta_v_thresholds * momentum + v_thresholds_grad
        rbm.delta_h_thresholds = rbm.delta_h_thresholds * momentum + h_thresholds_grad

        rbm.weights -= rbm.delta_weights
        rbm.v -= rbm.delta_v_thresholds
        rbm.h -= rbm.delta_h_thresholds
        vis_part_loss = ((v1 - v0) ** 2).sum()
        hid_part_loss = ((h1 - h0) ** 2).sum()
        return vis_part_loss, hid_part_loss

    def get_data_for_specific_rbm(self, original_data, layer_index):
        current_layer_index = 0
        data = original_data
        while current_layer_index < layer_index:
            data, _ = self.rbm_stack[current_layer_index].visible_to_hidden(data)
            if len(self.layers[current_layer_index]) == 3:
                post_processing_actions = self.layers[current_layer_index][2]
                for action in post_processing_actions:
                    if not isinstance(action, torch.nn.Dropout) and not isinstance(action, torch.nn.BatchNorm2d):
                        data = action(data)
            current_layer_index += 1
        return data

    def train_rbm_per_layer(self, loaders, device, rbm, pretrain_type, layer_index):
        train_from_batch_func = self.train_rbm_from_batch if isinstance(rbm, RBM) else self.train_crbm_from_batch
        losses = []
        val_fail_counter = 0
        early_stop = False
        prev_val_loss = 1e100
        epoch = 0
        train_loader = loaders["train_loader"]
        prev_loss = 1e100
        while epoch < Config.pretraining_epochs:
        # while not early_stop: #and epoch < Config.pretraining_epochs:
            loss = 0
            momentum = Config.momentum_beg if epoch < Config.momentum_change_epoch else Config.momentum_end
            for i, data in enumerate(train_loader):
                inputs = self.get_data_for_specific_rbm(data[0].to(device), layer_index)
                vis_loss, hid_loss = train_from_batch_func(rbm, inputs, pretrain_type, momentum)
                loss += vis_loss.item() + hid_loss.item()
            losses.append(loss)
            print(loss)
            if math.fabs(prev_loss - loss) < 1e-3:
                early_stop = True
            prev_loss = loss
            if Config.use_validation_dataset and epoch % Config.validate_every_epochs == 0:
                val_loader = loaders["val_loader"]
                val_loss = self.test_rbm(rbm, val_loader, device, layer_index)
                val_fail_counter = val_fail_counter + 1 if val_loss > prev_val_loss else 0
                if val_fail_counter == Config.validation_decay:
                    early_stop = True
                print("val_loss " + str(val_loss))
                prev_val_loss = val_loss
            epoch += 1
        return losses

    def test_rbm(self, rbm_model, val_loader, device, layer_index):
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs = self.get_data_for_specific_rbm(data[0].to(device), layer_index)
                v0, v1, _, h0, _, h1, _ = rbm_model(inputs)
                test_loss += ((v1 - v0) ** 2).sum().item() + ((h1 - h0) ** 2).sum().item()
        return test_loss

    def do_reduction(self, layers_config):
        masks = []
        # coef = 1e-3
        with torch.no_grad():
            for i in range(0, len(self.layers) - 1):
                pruning_param = torch.std(self.rbm_stack[i].weights)
                print(pruning_param)
                mask = torch.abs(self.rbm_stack[i].weights) > pruning_param * 0.1
                print(mask.shape)
                reduction_params_percent = (~mask).sum() * 100. / mask.numel()
                print(reduction_params_percent)
                self.rbm_stack[i].weights *= mask.double()
                if len(mask.shape) == 2:
                    mask = mask.T
                masks.append(mask)
                # coef *= 10
        return masks
            # condition = None
            # for i in range(0, len(self.layers) - 1):
            #     if condition is not None:
            #         self.rbm_stack[i].weights = nn.Parameter(self.rbm_stack[i].weights[:, condition])
            #     print(self.rbm_stack[i].weights.shape)
            #     if len(self.rbm_stack[i].weights.shape) > 2:
            #         condition = torch.abs(self.rbm_stack[i].weights).sum(dim=2).sum(dim=2).sum(dim=1) != 0
            #     else:
            #         condition = torch.abs(self.rbm_stack[i].weights).sum(dim=1) != 0
            #     print(condition.shape)
            #     print(condition)
            #     removed_columns_indices = torch.where(~condition)[0]
            #     print(removed_columns_indices)
            #     self.rbm_stack[i].weights = nn.Parameter(self.rbm_stack[i].weights[condition])
            #     self.rbm_stack[i].h = nn.Parameter(self.rbm_stack[i].h[:, condition])
                # layers_config["architecture"][i][0] = tuple(self.rbm_stack[i].weights.shape)
            # previous_neurons_count = layers_config["architecture"][len(self.layers)-2][0][1]
            # end_layer_neurons_count = layers_config["architecture"][len(self.layers)-1][0][1]
            # layers_config["architecture"][len(self.layers)-1][0] = (previous_neurons_count, end_layer_neurons_count)
