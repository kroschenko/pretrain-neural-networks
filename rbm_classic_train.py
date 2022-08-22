import torch
import config
import data_config
import utilities as utl
from matplotlib import pyplot as plt
from dataclasses import dataclass

import neptune.new as neptune

run = neptune.init(
    project="kroschenko/pretrain-networks",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NDM2YWM0Yy1iMGIxLTQwZTctYjMwNy04YWFiY2QxZjgzOWEifQ==",
)


DATASETS = [utl.DatasetType.MNIST, utl.DatasetType.CIFAR10]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_experiment_params():
    random_seeds = utl.get_random_seeds(config.count_attempts_in_experiment)
    layers_variants = config.get_layers_config_for_dataset(config.current_experiment_dataset_name).values()
    return random_seeds, layers_variants


@dataclass
class Conditions:
  layers: list
  rbm_variant: utl.PretrainingType
  dataset: utl.DatasetType
  
  def __str__(self) -> str:
      return str(self.dataset.name) + '/' +\
             str(self.layers) + '/' +\
             str(self.rbm_variant.name) + '/'             


for dataset in DATASETS:
    config.current_experiment_dataset_name = dataset
    trainset, testset, trainloader, testloader = data_config.get_torch_dataset(config.current_experiment_dataset_name, config.pretraining_batch_size)
    random_seeds, layers_variants = get_experiment_params()
    for layers in layers_variants:
        stat = utl.Statistics()
        for attempt_index in range(0, config.count_attempts_in_experiment):
            conditions = Conditions(layers, utl.PretrainingType.RBMClassic, dataset) 
            torch.random.manual_seed(random_seeds[attempt_index])
            statistics, losses = utl.run_experiment(run, layers, utl.PretrainingType.RBMClassic, trainset, trainloader, testloader, device)
            figure, ax = plt.subplots(1, 1, figsize=(10, 10))
            print(losses)
            ax.plot(losses)
            run[str(conditions)+str(attempt_index)+"/finetune_loss_evolution"].upload(figure)
            stat.add(statistics)
        run[str(conditions)+"mean_statistics"] = stat.get_mean_statistics()
    
    for layers in layers_variants:
        stat = utl.Statistics()
        for attempt_index in range(0, config.count_attempts_in_experiment):
            conditions = Conditions(layers, utl.PretrainingType.REBA, dataset) 
            torch.random.manual_seed(random_seeds[attempt_index])
            statistics, losses = utl.run_experiment(run, layers, utl.PretrainingType.REBA, trainset, trainloader, testloader, device)
            figure, ax = plt.subplots(1, 1, figsize=(10, 10))
            print(losses)
            ax.plot(losses)
            run[str(conditions)+str(attempt_index)+"/finetune_loss_evolution"].upload(figure)
            stat.add(statistics)
        run[str(conditions)+"mean_statistics"] = stat.get_mean_statistics()

run.stop()
