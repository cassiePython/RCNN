from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument('--total_epoch', type=int, default=100, help='total epoch for training')
        self._parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
        self._parser.add_argument('--decay_rate', type=float, default=0.99, help='decay rate')
        self._parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        
        self.is_train = True
