from __future__ import division
from data.dataset_factory import DatasetFactory
from models.model_factory import ModelsFactory
from options.train_options import TrainOptions
import numpy as np

class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()

        self._dataset_train = DatasetFactory.get_by_name("SVMDataset", self._opt)
        self._dataset_train_size = len(self._dataset_train)
        print('#train images = %d' % self._dataset_train_size)

        self.classA_features, self.classA_labels, self.classB_features, self.classB_labels = self._dataset_train.get_datas()

        self._modelA = ModelsFactory.get_by_name("SvmModel", self._opt, is_train=True)
        self._modelB = ModelsFactory.get_by_name("SvmModel", self._opt, is_train=True)

        self._train(self._modelA, self.classA_features, self.classA_labels, "A")
        self._train(self._modelB, self.classB_features, self.classB_labels, "B")

    def _train(self, model, features, labels, name):
        model.train(features, labels)
        model.save(name)
        pred = model.predict(features)

        print (labels)
        print (pred)
        
        

if __name__ == "__main__":
    Train()  

        
