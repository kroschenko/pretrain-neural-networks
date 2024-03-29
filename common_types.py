import enum


class DatasetType(enum.Enum):
    MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3
    IRIS = 4
    FashionMNIST = 5


class PretrainingType(enum.Enum):
    Hybrid = 1
    REBA = 2
    RBMClassic = 3
    Without = 4


class LayerTrainType(enum.Enum):
    PerLayer = 1
    PerBatch = 2
    PerBatchRandom = 3


class InitTypes(enum.Enum):
    SimpleUniform = 1
    SimpleNormal = 2
    Kaiming = 3


class Statistics:

    def __init__(self):
        self.total_acc = 0
        self.total_acc_lst = []

    def add(self, other):
        self.total_acc += other.total_acc
        self.total_acc_lst.append(other.total_acc)

    def get_mean_statistics(self):
        result = Statistics()
        result.total_acc = self.total_acc / len(self.total_acc_lst)
        print(self.total_acc_lst)
        return result

    @staticmethod
    def get_train_statistics(losses, total_acc):
        stat = Statistics()
        stat.total_acc = total_acc
        stat.losses = losses
        return stat

    def __str__(self):
        res = "Best total acc: " + str(self.total_acc) + "\n"
        return res
