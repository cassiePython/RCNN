from .model_factory import BaseModel
from networks.network_factory import NetworksFactory
from torch.autograd import Variable
from sklearn import svm
from collections import OrderedDict
import torch
import os
from sklearn.externals import joblib
from loss.losses import Regloss

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexModel(BaseModel):
    def __init__(self, opt, is_train):
        super(AlexModel, self).__init__(opt, is_train)
        self._name = 'AlexModel'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # use pre-trained AlexNet
        if self._is_train and not self._opt.load_alex_epoch > 0:
            self._init_weights()

        # load networks and optimizers
        if not self._is_train or self._opt.load_alex_epoch > 0:
            self.load()

        if not self._is_train:
            self.set_eval()

        # init loss
        self._init_losses()

    def _init_create_networks(self):
        network_type = 'AlexNet'
        self.network = self._create_branch(network_type)
        if len(self._gpu_ids) > 1:
            self.network = torch.nn.DataParallel(self.network, device_ids=self._gpu_ids)
        if torch.cuda.is_available():
            self.network.cuda()

    def _init_train_vars(self):
        self._current_lr = self._opt.learning_rate
        self._decay_rate = self._opt.decay_rate
        # initialize optimizers
        self._optimizer = torch.optim.SGD(self.network.parameters(),
                                         lr=self._current_lr,
                                         momentum=self._decay_rate)

    def _init_weights(self):
        state_dict=load_state_dict_from_url(model_urls['alexnet'], progress=True)
        current_state = self.network.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('features'):
                    current_state[key] = state_dict[key]
        current_state['fn8.weight'] = state_dict['classifier.1.weight']
        current_state['fn8.bias'] = state_dict['classifier.1.bias']
        current_state['fn9.weight'] = state_dict['classifier.4.weight']
        current_state['fn9.bias'] = state_dict['classifier.4.bias']

    def load(self):
        load_epoch = self._opt.load_alex_epoch
        self._load_network(self.network, 'AlexNet', load_epoch, self._opt.alex_dir)

    def save(self, label):
        # save networks
        self._save_network(self.network, 'AlexNet', label, self._opt.alex_dir)

    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name, self._opt.alex_classes)

    def _init_losses(self):
        # define loss function
        self._cross_entropy = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._cross_entropy = self._cross_entropy.cuda()

    def set_input(self, inputs, labels):
        self.batch_inputs = self._Tensor(inputs).permute(0,3,1,2)
        self.labels = Variable(self._LongTensor(labels).squeeze_())

    def optimize_parameters(self):
        if self._is_train:

            loss = self._forward()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _forward(self):
        feature, final = self.network(self.batch_inputs)
        self._loss = self._cross_entropy(final, self.labels)
        return self._loss

    def _forward_test(self, input):
        feature, final = self.network(input)
        return feature, final

    def get_current_errors(self):
        loss_dict = OrderedDict([('loss_entropy', self._loss.data[0]),
                                 ])
        return loss_dict

    def set_train(self):
        self.network.train()
        self._is_train = True

    def set_eval(self):
        self.network.eval()
        self._is_train = False


class FineModel(BaseModel):
    def __init__(self, opt, is_train):
        super(FineModel, self).__init__(opt, is_train)
        self._name = 'FineModel'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # use pre-trained AlexNet
        if self._is_train and not self._opt.load_finetune_epoch > 0:
            self._init_weights()

        # load networks and optimizers
        if not self._is_train or self._opt.load_finetune_epoch > 0:
            self.load()

        if not self._is_train:
            self.set_eval()

        # init loss
        self._init_losses()

    def _init_create_networks(self):
        network_type = 'AlexNet'
        self.network = self._create_branch(network_type)
        if len(self._gpu_ids) > 1:
            self.network = torch.nn.DataParallel(self.network, device_ids=self._gpu_ids)
        if torch.cuda.is_available():
            self.network.cuda()

    def _init_train_vars(self):
        self._current_lr = self._opt.learning_rate
        self._decay_rate = self._opt.decay_rate
        # initialize optimizers
        self._optimizer = torch.optim.SGD(self.network.parameters(),
                                         lr=self._current_lr,
                                         momentum=self._decay_rate)

    def _init_weights(self):

        load_epoch = self._opt.load_alex_epoch
        load_dir = self._opt.alex_dir
        network_label = "AlexNet"
        load_filename = 'net_epoch_%s_id_%s.pth' % (load_epoch, network_label)
        
        load_path = os.path.join(self._opt.checkpoints_dir, load_dir, load_filename)
        assert os.path.exists(
                load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        state_dict = torch.load(load_path)              
        current_state = self.network.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if not key.startswith('fn10'):
                current_state[key] = state_dict[key]

    def save(self, label):
        # save networks
        self._save_network(self.network, 'AlexNet', label, self._opt.finetune_dir)

    def load(self):
        load_epoch = self._opt.load_finetune_epoch
        self._load_network(self.network, 'AlexNet', load_epoch, self._opt.finetune_dir)

    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name, self._opt.finetune_classes)

    def _init_losses(self):
        # define loss function
        self._cross_entropy = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._cross_entropy = self._cross_entropy.cuda()

    def set_input(self, inputs, labels):
        self.batch_inputs = self._Tensor(inputs).permute(0,3,1,2)
        self.labels = Variable(self._LongTensor(labels).squeeze_())

    def optimize_parameters(self):
        if self._is_train:

            loss = self._forward()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _forward(self):
        feature, final = self.network(self.batch_inputs)
        self._loss = self._cross_entropy(final, self.labels)
        return self._loss

    def _forward_test(self, input):
        feature, final = self.network(input)
        return feature, final

    def get_current_errors(self):
        loss_dict = OrderedDict([('loss_entropy', self._loss.data[0]),
                                 ])
        return loss_dict

    def set_train(self):
        self.network.train()
        self._is_train = True

    def set_eval(self):
        self.network.eval()
        self._is_train = False


class SvmModel:
    def __init__(self, opt, is_train):
        self._opt = opt
        self._name = 'SvmModel'
        self._save_dir = os.path.join(opt.checkpoints_dir, opt.svm_dir)
        self._is_train = is_train
        
        
    def train(self, features, labels):
        self.clf = svm.LinearSVC()
        self.clf.fit(features, labels)

    def save(self, name):
        save_path = os.path.join(self._save_dir, "%s_svm.pkl" % name)
        joblib.dump(self.clf, save_path)

    def load(self, name):
        load_path = os.path.join(self._save_dir, "%s_svm.pkl" % name)
        self.clf = joblib.load(load_path)

    def predict(self, features):
        pred = self.clf.predict(features)
        return pred

    @property
    def name(self):
        return self._name

class RegModel(BaseModel):
    def __init__(self, opt, is_train):
        super(RegModel, self).__init__(opt, is_train)
        self._name = 'RegModel'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train and not self._opt.load_reg_epoch > 0:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_reg_epoch > 0:
            self.load()

        if not self._is_train:
            self.set_eval()

        # init loss
        self._init_losses()

    def _init_create_networks(self):
        self.network = self._create_branch("RegNet")
        if len(self._gpu_ids) > 1:
            self.network = torch.nn.DataParallel(self.network, device_ids=self._gpu_ids)
        if torch.cuda.is_available():
            self.network.cuda()

    def _init_train_vars(self):
        self._current_lr = self._opt.learning_rate
        self._decay_rate = self._opt.decay_rate
        # initialize optimizers
        self._optimizer = torch.optim.SGD(self.network.parameters(),
                                         lr=self._current_lr,
                                         momentum=self._decay_rate)

    def load(self):
        load_epoch = self._opt.load_reg_epoch    
        self._load_network(self.network, 'RegNet', load_epoch, self._opt.reg_dir)
            

    def save(self, label):
        # save networks
        self._save_network(self.network, 'RegNet', label, self._opt.reg_dir)

    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name, self._opt.reg_classes)

    def _init_losses(self):
        # define loss function
        self._reg_loss = Regloss()
        if torch.cuda.is_available():
            self._reg_loss = self._reg_loss.cuda()

    def set_input(self, inputs, labels):
        self.batch_inputs = self._Tensor(inputs)
        self.labels = Variable(self._FloatTensor(labels))

    def optimize_parameters(self):
        if self._is_train:

            loss = self._forward()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _forward(self):
        final = self.network(self.batch_inputs)
        self._loss = self._reg_loss(self.labels, final)
        return self._loss

    def _forward_test(self, input):
        final = self.network(input)
        return final

    def get_current_errors(self):
        loss_dict = OrderedDict([('loss_reg', self._loss.data[0]),
                                 ])
        return loss_dict

    def set_train(self):
        self.network.train()
        self._is_train = True

    def set_eval(self):
        self.network.eval()
        self._is_train = False

