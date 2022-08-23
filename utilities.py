import random
import config
import enum
import torchvision.datasets as datasets
import torch
from rbm_stack import RBMStack
from models import UnifiedClassifier
from torch import nn
import torch.optim as optim
import data_config
from torch.functional import F


class DatasetType(enum.Enum):
    MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3


class PretrainingType(enum.Enum):
    RBMClassic = 1
    REBA = 2


class Statistics:
    total_summed_statistics = 0

    def __init__(self):
        self.best_total_acc = 0
        self.best_total_acc_lst = []

    def add(self, other):
        self.best_total_acc += other.best_total_acc
        self.best_total_acc_lst.append(other.best_total_acc)
        Statistics.total_summed_statistics += 1

    def get_mean_statistics(self):
        result = Statistics()
        result.best_total_acc = self.best_total_acc / Statistics.total_summed_statistics
        Statistics.total_summed_statistics = 0
        print(self.best_total_acc_lst)
        return result

    @staticmethod
    def get_train_statistics(model, losses, best_total_acc):
        stat = Statistics()
        stat.best_total_acc = best_total_acc
        stat.losses = losses
        return stat

    def __str__(self):
        res = "Best total acc: " + str(self.best_total_acc) + "\n"
        return res


def get_random_seeds(count):
    seeds = []
    for i in range(0, count):
        seeds.append(random.randint(0, config.max_random_seed))
    return seeds


def get_dataset_constructor(dataset_name):
    dataset_selector = {DatasetType.MNIST: datasets.MNIST, DatasetType.CIFAR10: datasets.CIFAR10, DatasetType.CIFAR100: datasets.CIFAR100}
    return dataset_selector[dataset_name]


def get_sample_from_loader(train_loader, device, rbm):
    delta_weights = torch.zeros(rbm.W.shape).to(device)
    delta_v_threshols = torch.zeros(rbm.v.shape).to(device)
    delta_h_threshols = torch.zeros(rbm.h.shape).to(device)
    losses = []
    for epoch in range(config.pretraining_epochs):
        loss = 0.0
        momentum = config.momentum_beg if epoch < config.momentum_change_epoch else config.momentum_end
        for i, data in enumerate(train_loader, 0):
            inputs = data[0].to(device)
            v0, v1, h0, h1 = rbm(inputs)
            delta_weights = delta_weights * momentum + config.pretraining_rate / config.pretraining_batch_size * (
                    torch.mm(v0.T, h0) - torch.mm(v1.T, h1))
            delta_v_threshols = delta_v_threshols * momentum + config.pretraining_rate / config.pretraining_batch_size * (v0.sum(0) - v1.sum(0))
            delta_h_threshols = delta_h_threshols * momentum + config.pretraining_rate / config.pretraining_batch_size * (h0.sum(0) - h1.sum(0))
            rbm.W += delta_weights
            rbm.v += delta_v_threshols
            rbm.h += delta_h_threshols
            loss += ((v1 - v0) ** 2).sum()
        losses.append(loss.item())
    return losses


def get_sample_from_loader_reba(train_loader, device, rbm):
    delta_weights = torch.zeros(rbm.W.shape).to(device)
    delta_v_threshols = torch.zeros(rbm.v.shape).to(device)
    delta_h_threshols = torch.zeros(rbm.h.shape).to(device)
    losses = []
    for epoch in range(config.pretraining_epochs):
        loss = 0.0
        momentum = config.momentum_beg if epoch < config.momentum_change_epoch else config.momentum_end
        for i, data in enumerate(train_loader, 0):
            inputs = data[0].to(device)
            v0, v1, h0, h1 = rbm(inputs)
            der_v = v1 * (1 - v1)
            der_h = h1 * (1 - h1)
            part_v = (v1 - v0) * der_v
            part_h = (h1 - h0) * der_h
            delta_weights = delta_weights * momentum + config.pretraining_rate_reba / config.pretraining_batch_size * (
                    torch.mm(part_v.T, h0) + torch.mm(v1.T, part_h))
            delta_v_threshols = delta_v_threshols * momentum + config.pretraining_rate_reba / config.pretraining_batch_size * part_v.sum(0)
            delta_h_threshols = delta_h_threshols * momentum + config.pretraining_rate_reba / config.pretraining_batch_size * part_h.sum(0)
            rbm.W -= delta_weights
            rbm.v -= delta_v_threshols
            rbm.h -= delta_h_threshols
            loss += ((v1 - v0) ** 2).sum()
        losses.append(loss.item())
    return losses


def get_sample_from_custom_dataset(train_set, device, rbm, batches_count):
    delta_weights = torch.zeros(rbm.W.shape).to(device)
    delta_v_threshols = torch.zeros(rbm.v.shape).to(device)
    delta_h_threshols = torch.zeros(rbm.h.shape).to(device)
    losses = []
    for epoch in range(config.pretraining_epochs):
        loss = 0.0
        i = 0
        momentum = config.momentum_beg if epoch < config.momentum_change_epoch else config.momentum_end
        while i < batches_count:
            inputs = train_set[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size].to(device)
            v0, v1, h0, h1 = rbm(inputs)
            delta_weights = delta_weights * momentum + config.pretraining_rate / config.pretraining_batch_size * (
                    torch.mm(v0.T, h0) - torch.mm(v1.T, h1))
            delta_v_threshols = delta_v_threshols * momentum + config.pretraining_rate / config.pretraining_batch_size * (v0.sum(0) - v1.sum(0))
            delta_h_threshols = delta_h_threshols * momentum + config.pretraining_rate / config.pretraining_batch_size * (h0.sum(0) - h1.sum(0))
            rbm.W += delta_weights
            rbm.v += delta_v_threshols
            rbm.h += delta_h_threshols
            loss += ((v1 - v0) ** 2).sum()
            i += 1
        losses.append(loss.item())
    return losses


def get_sample_from_custom_dataset_reba(train_set, device, rbm, batches_count):
    delta_weights = torch.zeros(rbm.W.shape).to(device)
    delta_v_threshols = torch.zeros(rbm.v.shape).to(device)
    delta_h_threshols = torch.zeros(rbm.h.shape).to(device)
    losses = []
    for epoch in range(config.pretraining_epochs):
        loss = 0.0
        i = 0
        momentum = config.momentum_beg if epoch < config.momentum_change_epoch else config.momentum_end
        while i < batches_count:
            inputs = train_set[i * config.pretraining_batch_size:(i + 1) * config.pretraining_batch_size].to(device)
            v0, v1, h0, h1 = rbm(inputs)
            der_v = v1 * (1 - v1)
            der_h = h1 * (1 - h1)
            part_v = (v1 - v0) * der_v
            part_h = (h1 - h0) * der_h
            delta_weights = delta_weights * momentum + config.pretraining_rate_reba / config.pretraining_batch_size * (
                    torch.mm(part_v.T, h0) + torch.mm(v1.T, part_h))
            delta_v_threshols = delta_v_threshols * momentum + config.pretraining_rate_reba / config.pretraining_batch_size * part_v.sum(0)
            delta_h_threshols = delta_h_threshols * momentum + config.pretraining_rate_reba / config.pretraining_batch_size * part_h.sum(0)
            rbm.W -= delta_weights
            rbm.v -= delta_v_threshols
            rbm.h -= delta_h_threshols
            loss += ((v1 - v0) ** 2).sum()
            i += 1
        losses.append(loss.item())
    return losses


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


def run_experiment(run, layers, current_experiment_dataset_name, pretrain_type, train_set, train_loader, test_loader, device):
    rbm_stack = RBMStack(layers, device)
    layers_losses = rbm_stack.train(run, train_set, train_loader, pretrain_type)

    classifier = UnifiedClassifier(layers).to(device)
    rbm_stack.torch_model_init_from_weights(classifier)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    train_set, test_set, train_loader, test_loader = data_config.get_torch_dataset(
        current_experiment_dataset_name,
        config.finetuning_batch_size
    )
    best_total_acc, losses = train_torch_model(classifier, train_loader, test_loader, optimizer, criterion, device)

    return Statistics.get_train_statistics(classifier, layers_losses, best_total_acc), losses
