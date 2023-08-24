import random
from config import Config
import torch
from rbm_stack import RBMStack
from models import UnifiedClassifier
from torch import nn
import torch.optim as optim
from common_types import PretrainingType, Statistics
from torch.optim.lr_scheduler import StepLR
import data_config


def get_random_seeds(count):
    seeds = []
    for i in range(0, count):
        seeds.append(random.randint(0, Config.max_random_seed))
    return seeds


def test_rbm(rbm_model, val_loader, device):
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs = data[0].to(device)
            v0, v1, _, _, _, _, _ = rbm_model(inputs)
            test_loss += ((v1 - v0) ** 2).sum().item()
    return test_loss


def train_torch_model(model, loaders, optimizer, criterion, scheduler, device):
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
        scheduler.step()
        if epoch % Config.test_every_epochs == 0:
            current_accuracy, test_loss = test_torch_model(model, test_loader, criterion, device)
            if current_accuracy > best_total_accuracy:
                best_total_accuracy = current_accuracy
        if Config.use_validation_dataset and epoch % Config.validate_every_epochs == 0:
            current_accuracy, val_loss = test_torch_model(model, val_loader, criterion, device)
            val_fail_counter = val_fail_counter + 1 if val_loss > prev_val_loss else 0
            if val_fail_counter == Config.validation_decay:
                early_stop = True
            prev_val_loss = val_loss
            print("val loss = " + str(val_loss.item()) + " val_accuracy = " + str(current_accuracy))
        losses.append(running_loss)
        epoch += 1
    return best_total_accuracy, losses


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


def run_experiment(layers_config, pretrain_type, loaders, device, init_type, without_sampling):
    rbm_stack = RBMStack(layers_config, device, init_type, without_sampling)
    layers_losses = None
    if pretrain_type != PretrainingType.Without:
        layers_losses = rbm_stack.train(loaders, pretrain_type, layer_train_type=Config.layer_train_type)
        if Config.with_reduction:
            rbm_stack.do_reduction(layers_config)

    classifier = UnifiedClassifier(layers_config).to(device)
    rbm_stack.torch_model_init_from_weights(classifier)

    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss(reduction="sum")
    # optimizer = optim.SGD(
    # classifier.parameters(), lr=config.finetune_rate, momentum=config.fine_tuning_momentum, weight_decay=1e-6
    # )
    optimizer = optim.Adam(classifier.parameters(), lr=Config.finetune_rate, weight_decay=1e-6)
    scheduler = StepLR(optimizer, 5, 0.1)
    # loaders["train_loader"].dataset.transform = data_config.transform_COMMON
    best_total_acc, losses = train_torch_model(classifier, loaders, optimizer, criterion, scheduler, device)

    return Statistics.get_train_statistics(layers_losses, best_total_acc), losses
