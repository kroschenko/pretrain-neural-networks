import utilities as utl

max_random_seed = 1024

pretraining_batch_size = 128
momentum_beg = 0.5
momentum_end = 0.9
momentum_change_epoch = 5
pretraining_epochs = 10
pretraining_rate = 0.01
pretraining_rate_reba = 0.04

finetune_rate = 0.1
finetuning_batch_size = 128
finetuning_epochs = 10
finetuning_momentum = 0.9
test_every_epochs = 5
count_attempts_in_experiment = 2


def get_layers_config_for_dataset(experiment_dataset_name):
    if experiment_dataset_name == utl.DatasetType.MNIST:
        return {"var1": [784, 800, 800, 10], "var2": [784, 1600, 1600, 800, 800, 10]}
    elif experiment_dataset_name == utl.DatasetType.CIFAR10:
        return {"var1": [3072, 1024, 512, 256, 128, 64, 10], "var2": [3072, 512, 256, 128, 64, 10]}
    elif experiment_dataset_name == utl.DatasetType.CIFAR100:
        return {"var1": [3072, 1536, 1024, 512, 256, 100]}
    else:
        return None
