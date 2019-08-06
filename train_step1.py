from __future__ import division
from data.dataset_factory import DatasetFactory
from models.model_factory import ModelsFactory
from options.train_options import TrainOptions

class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()

        self._dataset_train = DatasetFactory.get_by_name("AlexnetDataset", self._opt)
        self._dataset_train_size = len(self._dataset_train)
        print('#train images = %d' % self._dataset_train_size)

        self._model = ModelsFactory.get_by_name("AlexModel", self._opt, is_train=True)

        self._train()

    def _train(self):
        self._steps_per_epoch = int (self._dataset_train_size / self._opt.batch_size)
        
        for i_epoch in range(self._opt.load_alex_epoch + 1, self._opt.total_epoch + 1):
            # train epoch
            self._train_epoch(i_epoch)

            # save model
            if i_epoch % 20 == 0:
                print('saving the model at the end of epoch %d' % i_epoch)
                self._model.save(i_epoch)

    def _train_epoch(self, i_epoch):

        for step in range(1, self._steps_per_epoch+1):
            input, labels = self._dataset_train.get_batch()

            # train model
            self._model.set_input(input, labels)
            self._model.optimize_parameters()

            # display terminal
            self._display_terminal_train(i_epoch, step)

    def _display_terminal_train(self, i_epoch, i_train_batch):
        errors = self._model.get_current_errors()
        message = '(epoch: %d, it: %d/%d) ' % (i_epoch, i_train_batch, self._steps_per_epoch)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)

if __name__ == "__main__":
    Train()
        

            
            
            
        
