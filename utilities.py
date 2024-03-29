import random

import config
import numpy as np
from config import Config
import torch
from rbm_stack import RBMStack
from models import UnifiedClassifier
from torch import nn
import torch.optim as optim
from common_types import PretrainingType, Statistics
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F


def get_random_seeds(count):
    seeds = []
    for i in range(0, count):
        seeds.append(random.randint(0, Config.max_random_seed))
    return seeds


def train_torch_model(model, loaders, optimizer, criterion, device, masks=None, with_reduction=False):
    best_total_accuracy = 0
    losses = []
    train_loader = loaders["train_loader"]
    test_loader = loaders["test_loader"]
    val_loader = loaders["val_loader"]
    val_fail_counter = 0
    epoch = 0
    early_stop = False
    prev_val_loss = 1e100
    while epoch < Config.max_finetuning_epochs:
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if masks is not None:
                zeroing_parameters(model, masks)
        # scheduler.step()
        if epoch % Config.test_every_epochs == 0:
            current_accuracy, test_loss = test_torch_model(model, test_loader, criterion, device)
            if current_accuracy > best_total_accuracy:
                best_total_accuracy = current_accuracy
            print("test loss = " + str(test_loss.item()) + " test_accuracy = " + str(current_accuracy))
        if Config.use_validation_dataset and epoch % Config.validate_every_epochs == 0:
            current_accuracy, val_loss = test_torch_model(model, val_loader, criterion, device)
            val_fail_counter = val_fail_counter + 1 if val_loss > prev_val_loss else 0
            if val_fail_counter == Config.validation_decay:
                early_stop = True
            prev_val_loss = val_loss
            print("val loss = " + str(val_loss.item()) + " val_accuracy = " + str(current_accuracy))
        print(running_loss)
        losses.append(running_loss)
        epoch += 1
        if with_reduction and epoch == Config.reduction_after_epochs and masks is None:
            masks = get_masks_for_zeroing(model)
    return best_total_accuracy, losses


def get_masks_for_zeroing(model):
    masks = []
    with torch.no_grad():
        for layer_num in range(0, len(model.layers)-1):
            pruning_param = torch.std(model.layers[layer_num].weight)
            print(pruning_param)
            mask = torch.abs(model.layers[layer_num].weight) > pruning_param * 0.1
            print(mask.shape)
            reduction_params_percent = (~mask).sum() * 100. / mask.numel()
            print(reduction_params_percent)
            model.layers[layer_num].weight *= mask.double()
            masks.append(mask)
    return masks


def zeroing_parameters(model, masks):
    with torch.no_grad():
        for layer_num in range(0, len(masks)):
            # print(masks[layer_num].shape)
            # print(model.layers[layer_num].weight.shape)
            model.layers[layer_num].weight *= masks[layer_num]
    # for name, param in model.named_parameters():

        # print(name, param.size())


def test_torch_model(model, test_loader, criterion, device):
    correct_answers = 0
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            correct_answers += (predictions == labels).sum()
    return 100 * float(correct_answers) / len(test_loader.dataset), test_loss


def run_experiment(layers_config, pretrain_type, loaders, device, init_type, without_sampling, with_reduction):
    rbm_stack = RBMStack(layers_config, device, init_type, without_sampling)
    layers_losses = None
    masks = None
    if pretrain_type != PretrainingType.Without:
        layers_losses = rbm_stack.train(loaders, pretrain_type, layer_train_type=Config.layer_train_type)
        if with_reduction:
            masks = rbm_stack.do_reduction(layers_config)
    else:
        if not with_reduction:
            Config.max_finetuning_epochs += Config.pretraining_epochs
    # print(len(masks))
    classifier = UnifiedClassifier(layers_config).to(device)
    rbm_stack.torch_model_init_from_weights(classifier)

    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.SGD(
        classifier.parameters(), lr=Config.finetune_rate, momentum=Config.finetuning_momentum, weight_decay=1e-6
    )
    # optimizer = optim.Adam(classifier.parameters(), lr=Config.finetune_rate, weight_decay=1e-6)
    # scheduler = StepLR(optimizer, 10, 0.5)
    # loaders["train_loader"].dataset.transform = data_config.transform_COMMON
    best_total_acc, losses = train_torch_model(classifier, loaders, optimizer, criterion, device, masks, with_reduction)

    return Statistics.get_train_statistics(layers_losses, best_total_acc), losses
