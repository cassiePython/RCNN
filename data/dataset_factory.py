import os

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt):
        if dataset_name == 'AlexnetDataset':
            from data.datasets import AlexnetDataset
            dataset = AlexnetDataset(opt)
        elif dataset_name == 'FinetuneDataset':
            from data.datasets import FinetuneDataset
            dataset = FinetuneDataset(opt)
        elif dataset_name == 'SVMDataset':
            from data.datasets import SVMDataset
            dataset = SVMDataset(opt)
        elif dataset_name == 'RegDataset':
            from data.datasets import RegDataset
            dataset = RegDataset(opt)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset

class DatasetBase:
    def __init__(self, opt):
        self._name = 'BaseDataset'
        self._root = opt.data_dir
        self._opt = opt

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root
