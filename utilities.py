import random
from config import Config
import torch
from rbm_stack import RBMStack
from models import UnifiedClassifier
from torch import nn
import torch.optim as optim
from common_types import PretrainingType, Statistics
from models import RBM
from torch.optim.lr_scheduler import StepLR


def get_random_seeds(count):
    seeds = []
    for i in range(0, count):
        seeds.append(random.randint(0, Config.max_random_seed))
    return seeds


def train_rbm_with_custom_dataset(train_set, device, rbm, pretrain_type, batches_count):
    train_func = train_rbm if isinstance(rbm, RBM) else train_crbm
    losses, output_shape = train_func(rbm, device, batches_count, train_set, pretrain_type)
    return output_shape


def train_rbm(rbm, device, batches_count, train_set, pretrain_type):
    delta_weights = torch.zeros(rbm.weights.shape).to(device)
    delta_v_thresholds = torch.zeros(rbm.v.shape).to(device)
    delta_h_thresholds = torch.zeros(rbm.h.shape).to(device)
    losses = []
    act_func = rbm.a_func
    for epoch in range(Config.pretraining_epochs):
        rand_indx = torch.randperm(len(train_set))
        train_set = train_set[rand_indx]
        loss = 0.
        i = 0
        momentum = Config.momentum_beg if epoch < Config.momentum_change_epoch else Config.momentum_end
        while i < batches_count:
            inputs = train_set[i * Config.pretraining_batch_size:(i + 1) * Config.pretraining_batch_size].to(device)
            v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = rbm(inputs)
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
                rate = (((v0-v1) * b_v).sum() + ((h0-h1) * b_h).sum()) / ((b_h*b_h).sum() + (b_v*b_v).sum())
            part_v = (v1 - v0) * der_v
            part_h = (h1 - h0) * der_h
            delta_weights = delta_weights * momentum + rate / Config.pretraining_batch_size * (
                    torch.mm(part_v.T, h0) + torch.mm(v1.T, part_h))
            delta_v_thresholds = delta_v_thresholds * momentum + rate / Config.pretraining_batch_size * part_v.sum(0)
            delta_h_thresholds = delta_h_thresholds * momentum + rate / Config.pretraining_batch_size * part_h.sum(0)
            rbm.weights -= delta_weights
            rbm.v -= delta_v_thresholds
            rbm.h -= delta_h_thresholds
            part_loss = ((v1 - v0) ** 2).sum()
            loss += part_loss.item()
            i += 1
        print(loss)
        losses.append(loss)
    return losses, h0.shape


def train_crbm(rbm, device, batches_count, train_set, pretrain_type):
    delta_weights = torch.zeros(rbm.weights.shape).to(device)
    delta_v_thresholds = torch.zeros(rbm.v.shape).to(device)
    delta_h_thresholds = torch.zeros(rbm.h.shape).to(device)
    losses = []
    act_func = rbm.a_func
    for epoch in range(Config.pretraining_epochs):
        rand_indx = torch.randperm(len(train_set))
        train_set = train_set[rand_indx]
        loss = 0.0
        i = 0
        momentum = Config.momentum_beg if epoch < Config.momentum_change_epoch else Config.momentum_end
        while i < batches_count:
            inputs = train_set[i * Config.pretraining_batch_size:(i + 1) * Config.pretraining_batch_size].to(device)
            v0, v1, v1_ws, h0, h0_ws, h1, h1_ws = rbm(inputs)
            if pretrain_type == PretrainingType.RBMClassic:
                der_v, der_h = 1, 1
                rate = Config.pretraining_rate
            if pretrain_type == PretrainingType.REBA:
                der_v = (act_func[0](v1_ws+0.00001) - act_func[0](v1_ws-0.00001)) / 0.00002
                der_h = (act_func[1](h1_ws+0.00001) - act_func[1](h1_ws-0.00001)) / 0.00002
                rate = Config.pretraining_rate_reba
            part_v = (v1 - v0) * der_v
            part_h = (h1 - h0) * der_h
            first_convolution_part = torch.convolution(
                torch.permute(part_v, (1, 0, 2, 3)),
                torch.permute(h0, (1, 0, 2, 3)), None, stride=[1,1], padding=[0,0], dilation=[1,1], transposed=False, output_padding=[0,0], groups=1)
            second_convolution_part = torch.convolution(
                torch.permute(v1, (1, 0, 2, 3)),
                torch.permute(part_h, (1, 0, 2, 3)), None, stride=[1,1], padding=[0,0], dilation=[1,1], transposed=False, output_padding=[0,0], groups=1)
            common_conv_expr = first_convolution_part + second_convolution_part

            delta_weights = delta_weights * momentum + rate / Config.pretraining_batch_size * torch.permute(common_conv_expr, (1, 0, 2, 3))
            delta_v_thresholds = delta_v_thresholds * momentum + rate / (Config.pretraining_batch_size*part_v.shape[2]**2) * part_v.sum((0, 2, 3)).reshape(rbm.v.shape)
            delta_h_thresholds = delta_h_thresholds * momentum + rate / (Config.pretraining_batch_size*part_h.shape[2]**2) * part_h.sum((0, 2, 3)).reshape(rbm.h.shape)

            rbm.weights -= delta_weights
            rbm.v -= delta_v_thresholds
            rbm.h -= delta_h_thresholds
            loss += ((v1 - v0) ** 2).sum().item()
            i += 1
        print(loss)
        losses.append(loss)
    return losses, h0.shape


def train_torch_model(model, meta_data, optimizer, criterion, scheduler, device):
    best_total_accuracy = 0
    losses = []
    train_loader = meta_data[0]
    test_loader = meta_data[1]
    val_loader = meta_data[2]
    val_fail_counter = 0
    epoch = 0
    early_stop = False
    prev_val_loss = 1e100
    val_dataset_size = len(val_loader.dataset)
    while not early_stop and epoch < Config.max_finetuning_epochs:
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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
            # print(str(test_loss.item()) + " " + str(current_accuracy))
        if Config.use_validation_dataset and epoch % Config.validate_every_epochs == 0:
            current_accuracy, val_loss = test_torch_model(model, val_loader, criterion, device)
            val_fail_counter = val_fail_counter + 1 if val_loss > prev_val_loss else 0
            if val_fail_counter == Config.validation_decay:
                early_stop = True
            prev_val_loss = val_loss
            print("val loss = " + str(val_loss.item()) + " val_accuracy = " + str(current_accuracy))
        losses.append(running_loss)
        epoch += 1
        # print(running_loss)
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


def run_experiment(layers_config, pretrain_type, meta_data, device, init_type, without_sampling):
    rbm_stack = RBMStack(layers_config, device, init_type, without_sampling)
    layers_losses = None
    train_loader = meta_data[0]
    if pretrain_type != PretrainingType.Without:
        layers_losses = rbm_stack.train(train_loader, pretrain_type, layer_train_type=Config.layer_train_type)
        if Config.with_reduction:
            rbm_stack.do_reduction(layers_config)

    classifier = UnifiedClassifier(layers_config).to(device)
    rbm_stack.torch_model_init_from_weights(classifier)

    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss(reduction="sum")
    # optimizer = optim.SGD(classifier.parameters(), lr=config.finetune_rate, momentum=config.finetuning_momentum, weight_decay=1e-6)
    optimizer = optim.Adam(classifier.parameters(), lr=Config.finetune_rate, weight_decay=1e-6)
    scheduler = StepLR(optimizer, 10, 0.5)
    best_total_acc, losses = train_torch_model(classifier, meta_data, optimizer, criterion, scheduler, device)

    return Statistics.get_train_statistics(layers_losses, best_total_acc), losses
