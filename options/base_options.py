import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        
        self._parser.add_argument('--data_dir', type=str, default='sample_dataset', help='path to dataset')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--generate_save_path', type=str, default='./generate', help='dir to save generated data')
        
        self._parser.add_argument('--load_alex_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--load_finetune_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--load_reg_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')

        self._parser.add_argument('--alex_dir', type=str, default='alexnet', help='dir for training alexnet')
        self._parser.add_argument('--finetune_dir', type=str, default='finetune', help='dir for finetuning alexnet')
        self._parser.add_argument('--svm_dir', type=str, default='svm', help='dir for training svm')
        self._parser.add_argument('--reg_dir', type=str, default='regression', help='dir for box regression')
        self._parser.add_argument('--options_dir', type=str, default='alexnet', help='dir for saving options')

        self._parser.add_argument('--alex_classes', type=int, default=17, help='# classes when training alexnet')
        self._parser.add_argument('--finetune_classes', type=int, default=3, help='# classes when finetuning alexnet')
        self._parser.add_argument('--reg_classes', type=int, default=5, help='# classes when training regnet')
            
        self._parser.add_argument('--finetune_threshold', type=float, default=0.3, help='threshold to select image region')
        self._parser.add_argument('--svm_threshold', type=float, default=0.3, help='threshold to select image region')
        self._parser.add_argument('--reg_threshold', type=float, default=0.6, help='threshold to select image region')
        
        self._parser.add_argument('--train_list', type=str, default='train_list.txt', help='training data')            
        self._parser.add_argument('--finetune_list', type=str, default='fine_tune_list.txt', help='training data')
        
        self._parser.add_argument('--image_size', type=int, default=224, help='input image size')
        
        
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')       
        
        
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        # set gpu ids
        if torch.cuda.is_available() and len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.options_dir)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
