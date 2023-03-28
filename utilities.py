import random
import config
import torchvision.datasets as datasets
import torch
from rbm_stack import RBMStack
from models import UnifiedClassifier
from torch import nn
import torch.optim as optim
import data_config
from torch.functional import F
from common_types import PretrainingType, DatasetType, Statistics
from models import RBM


def get_random_seeds(count):
    seeds = []
    for i in range(0, count):
        seeds.append(random.randint(0, config.max_random_seed))
    return seeds


def get_dataset_constructor(dataset_type: DatasetType):
    dataset_selector = {
        DatasetType.MNIST: datasets.MNIST,
        DatasetType.CIFAR10: datasets.CIFAR10,
        DatasetType.CIFAR100: datasets.CIFAR100
    }
    return dataset_selector[dataset_type]


def train_rbm_with_custom_dataset(train_set, device, rbm, pretrain_type, batches_count):
    train_func = train_rbm if isinstance(rbm, RBM) else train_crbm
    output_shape = train_func(rbm, device, batches_count, train_set, pretrain_type)
    return output_shape


def train_rbm(rbm, device, batches_count, train_set, pretrain_type):
    delta_weights = torch.zeros(rbm.W.shape).to(device)
    delta_v_thresholds = torch.zeros(rbm.v.shape).to(device)
    delta_h_thresholds = torch.zeros(rbm.h.shape).to(device)
    losses = []
    act_func = rbm.a_func
    for epoch in range(config.pretraining_epochs):
        loss = 0.
        i = 0
        momentum = config.momentum_beg if epoch < config.momentum_change_epoch else config.momentum_end
        while i < batches_count:
            inputs = train_set[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size].to(device)
            v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = rbm(inputs)
            if pretrain_type == PretrainingType.RBMClassic:
                der_v, der_h = 1, 1
                rate = config.pretraining_rate
            elif pretrain_type == PretrainingType.REBA:
                der_v = (act_func(v1_ws + 0.00001) - act_func(v1_ws - 0.00001)) / 0.00002
                der_h = (act_func(h1_ws + 0.00001) - act_func(h1_ws - 0.00001)) / 0.00002
                rate = config.pretraining_rate_reba
            elif pretrain_type == PretrainingType.Hybrid:
                if epoch < 7:
                    der_v, der_h = 1, 1
                    rate = config.pretraining_rate
                else:
                    der_v = (act_func(v1_ws + 0.00001) - act_func(v1_ws - 0.00001)) / 0.00002
                    der_h = (act_func(h1_ws + 0.00001) - act_func(h1_ws - 0.00001)) / 0.00002
                    rate = config.pretraining_rate_reba
            part_v = (v1 - v0) * der_v
            part_h = (h1 - h0) * der_h
            delta_weights = delta_weights * momentum + rate / config.pretraining_batch_size * (
                    torch.mm(part_v.T, h0) + torch.mm(v1.T, part_h))
            delta_v_thresholds = delta_v_thresholds * momentum + rate / config.pretraining_batch_size * part_v.sum(0)
            delta_h_thresholds = delta_h_thresholds * momentum + rate / config.pretraining_batch_size * part_h.sum(0)
            rbm.W -= delta_weights
            rbm.v -= delta_v_thresholds
            rbm.h -= delta_h_thresholds
            part_loss = ((v1 - v0) ** 2).sum()
            loss += part_loss.item()
            i += 1
        print(loss)
        losses.append(loss)
    return losses, h0.shape


def train_crbm(rbm, device, batches_count, train_set, pretrain_type):
    delta_weights = torch.zeros(rbm.W.shape).to(device)
    losses = []
    act_func = rbm.a_func
    for epoch in range(config.pretraining_epochs):
        loss = 0.0
        i = 0
        momentum = config.momentum_beg if epoch < config.momentum_change_epoch else config.momentum_end
        while i < batches_count:
            inputs = train_set[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size].to(device)
            v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = rbm(inputs)
            if pretrain_type == PretrainingType.RBMClassic:
                der_v, der_h = 1, 1
                rate = config.pretraining_rate
            if pretrain_type == PretrainingType.REBA:
                der_v = (act_func(v1_ws+0.00001) - act_func(v1_ws-0.00001)) / 0.00002
                der_h = (act_func(h1_ws+0.00001) - act_func(h1_ws-0.00001)) / 0.00002
                rate = config.pretraining_rate_reba
            if pretrain_type == PretrainingType.Hybrid:
                if epoch < 7:
                    der_v, der_h = 1, 1
                    rate = config.pretraining_rate
                else:
                    der_v = (act_func(v1_ws + 0.00001) - act_func(v1_ws - 0.00001)) / 0.00002
                    der_h = (act_func(h1_ws + 0.00001) - act_func(h1_ws - 0.00001)) / 0.00002
                    rate = config.pretraining_rate_reba
            part_v = (v1 - v0) * der_v
            part_h = (h1 - h0) * der_h
            first_convolution_part = torch.convolution(
                torch.permute(part_v, (1, 0, 2, 3)),
                torch.permute(h0, (1, 0, 2, 3)), None, stride=[1,1], padding=[0,0], dilation=[1,1], transposed=False, output_padding=[0,0], groups=1)
            second_convolution_part = torch.convolution(
                torch.permute(v1, (1, 0, 2, 3)),
                torch.permute(part_h, (1, 0, 2, 3)), None, stride=[1,1], padding=[0,0], dilation=[1,1], transposed=False, output_padding=[0,0], groups=1)
            common_conv_expr = first_convolution_part + second_convolution_part

            delta_weights = delta_weights * momentum + rate / config.pretraining_batch_size * torch.permute(common_conv_expr, (1, 0, 2, 3))
            rbm.W -= delta_weights
            loss += ((v1 - v0) ** 2).sum()
            i += 1
        print(loss.item())
        losses.append(loss.item())
    return losses, h0.shape


def train_torch_model(model, train_loader, test_loader, optimizer, criterion, device):
    best_total_accuracy = 0
    losses = []
    for epoch in range(config.finetuning_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, F.one_hot(labels, num_classes=10).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % config.test_every_epochs == 0:
            current_accuracy = test_torch_model(model, test_loader, device)
            if current_accuracy > best_total_accuracy:
                best_total_accuracy = current_accuracy
        losses.append(running_loss)
        print(running_loss)
    return best_total_accuracy, losses


def test_torch_model(model, test_loader, device):
    correct_answers = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            correct_answers += (predictions == labels).sum()
    return 100 * float(correct_answers) / len(test_loader.dataset)


def run_experiment(layers_config, pretrain_type, meta_data, device, init_type, without_sampling):
    rbm_stack = RBMStack(layers_config, device, init_type, without_sampling)
    layers_losses = None
    train_loader = meta_data[2]
    train_set = data_config.get_tensor_dataset_from_loader(train_loader)
    if pretrain_type != PretrainingType.Without:
        layers_losses = rbm_stack.train(train_set, pretrain_type)

    classifier = UnifiedClassifier(layers_config).to(device)
    rbm_stack.torch_model_init_from_weights(classifier)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=config.finetune_rate, momentum=config.finetuning_momentum)

    test_loader = meta_data[3]
    best_total_acc, losses = train_torch_model(classifier, train_loader, test_loader, optimizer, criterion, device)

    return Statistics.get_train_statistics(layers_losses, best_total_acc), losses
