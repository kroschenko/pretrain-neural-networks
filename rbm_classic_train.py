import torch
import config
from config import ProjectConfig
import data_config
import utilities as utl
from matplotlib import pyplot as plt
from dataclasses import dataclass
from common_types import DatasetType, PretrainingType, Statistics

import neptune


def get_experiment_params(_current_experiment_dataset_name: DatasetType):
    _random_seeds = utl.get_random_seeds(config.count_attempts_in_experiment)
    _layers_variants = config.get_layers_config_for_dataset(_current_experiment_dataset_name)
    return _random_seeds, _layers_variants


@dataclass
class Conditions:
    layers: list
    rbm_pretraining: PretrainingType
    dataset: DatasetType

    def __str__(self) -> str:
        return str(self.dataset.name) + '/' + \
               str(self.layers) + '/' + \
               str(self.rbm_pretraining.name) + '/'


run = neptune.init_run(
    project=ProjectConfig.project_name,
    api_token=ProjectConfig.api_token
)

DATASETS = [DatasetType.MNIST]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for dataset in DATASETS:
    current_experiment_dataset_name = dataset
    meta_data = data_config.get_torch_dataset(current_experiment_dataset_name, config.pretraining_batch_size)
    random_seeds, layers_variants = get_experiment_params(current_experiment_dataset_name)
    conditions = "undefined_"
    pretraining_types = list(PretrainingType)
    for pretraining_type in pretraining_types:
        print(pretraining_type)
        for layers_config in layers_variants:
            stat = Statistics()
            for attempt_index in range(0, config.count_attempts_in_experiment):
                conditions = Conditions(layers_config, pretraining_type, dataset)
                torch.random.manual_seed(random_seeds[attempt_index])
                statistics, losses = utl.run_experiment(
                    layers_config, pretraining_type, meta_data, device, config.init_type, config.without_sampling
                )
                figure, ax = plt.subplots(1, 1, figsize=(10, 10))
                print(losses)
                ax.plot(losses)
                run[str(conditions) + str(attempt_index) + "/finetune_loss_evolution"].upload(figure)
                stat.add(statistics)
            print(stat.get_mean_statistics())
            run[str(conditions) + "mean_statistics"] = stat.get_mean_statistics()

run.stop()
