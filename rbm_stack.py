import torch
import utilities as utl
import config
from models import RBM



class RBMStack:
    def __init__(self, layers, device):
        self.rbm_stack = []
        self.layers = layers
        self.device = device
        for i in range(0, len(self.layers) - 1):
            rbm = RBM(self.layers[i], self.layers[i + 1])
            self.rbm_stack.append(rbm.to(self.device))


    def train(self, run, trainset, trainloader, pretrain_type):
        layers_losses = {}
        with torch.no_grad():
            rbm = self.rbm_stack[0]
            if pretrain_type == utl.PretrainingType.RBMClassic:
                layers_losses["layer_0"] = utl.get_sample_from_loader(trainloader, self.device, rbm)
            else:
                layers_losses["layer_0"] = utl.get_sample_from_loader_reba(trainloader, self.device, rbm)
            new_trainset = torch.zeros((trainset.data.shape[0], self.layers[1]))
            batches_count = len(new_trainset) / config.pretraining_batch_size
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                v0, v1, h0, h1 = rbm(inputs)
                new_trainset[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size] = h0
            j = 1
            for rbm, layer in zip(self.rbm_stack[1:-1], self.layers[2:-1]):
                if pretrain_type == utl.PretrainingType.RBMClassic:
                    layers_losses["layer_"+str(j)] = utl.get_sample_from_custom_dataset(new_trainset, self.device, rbm, batches_count)
                else:
                    layers_losses["layer_"+str(j)] = utl.get_sample_from_custom_dataset_reba(new_trainset, self.device, rbm, batches_count)
                new_trainset_next = torch.zeros((new_trainset.shape[0], layer))
                i = 0
                while i < batches_count:
                    inputs = new_trainset[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size].to(self.device)
                    v0, v1, h0, h1 = rbm(inputs)
                    new_trainset_next[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size] = h0
                    i += 1
                new_trainset = new_trainset_next
                j += 1
        return layers_losses

    def torch_model_init_from_weights(self, torch_model):
        with torch.no_grad():
            for i in range(0, len(torch_model.layers) - 1):
                torch_model.layers[i].weight.data = self.rbm_stack[i].W.T
                torch_model.layers[i].bias.data = torch.reshape(self.rbm_stack[i].h, (len(self.rbm_stack[i].h[0]),))
