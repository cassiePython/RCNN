from data.dataset_factory import DatasetBase
from utils.util import resize_image
from utils.util import clip_pic
from utils.util import IOU
from utils.util import if_intersection
from utils.util import image_proposal
from models.model_factory import ModelsFactory
from sklearn.externals import joblib
import numpy as np
import codecs
import cv2
import os
import torch

class AlexnetDataset(DatasetBase):

    def __init__(self, opt):
        super(AlexnetDataset, self).__init__(opt)
        self._name = 'AlexnetDataset'

        self.datas = []
        
        self._img_size = self._opt.image_size
        self._batch_size = self._opt.batch_size

        self.epoch = 0
        self.cursor = 0
        
        # read dataset
        self._load_dataset()

    def _load_dataset(self):
        file_path = os.path.join(self._root, self._opt.train_list)
        with codecs.open(file_path, 'r', 'utf-8') as fr:
            lines = fr.readlines()
            for ind, line in enumerate(lines):
                context = line.strip().split()
                image_path = os.path.join(self._root, context[0])
                label = int(context[1])

                img = cv2.imread(image_path)                        
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = resize_image(img, self._img_size, self._img_size)
                img = np.asarray(img, dtype='float32')

                self.datas.append([img, label])

    def get_batch(self):
        images = np.zeros((self._batch_size, self._img_size, self._img_size, 3))
        labels = np.zeros((self._batch_size, 1))
        count = 0
        while( count < self._batch_size):
            images[count] = self.datas[self.cursor][0]
            labels[count] = self.datas[self.cursor][1]
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.datas) :
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.datas)
        return images, labels

    def __len__(self):
        return len(self.datas)


class FinetuneDataset(DatasetBase):
    
    def __init__(self, opt):
        super(FinetuneDataset, self).__init__(opt)
        self._name = 'FinetuneDataset'

        self.datas = []
        self.save_path = os.path.join(self._opt.generate_save_path, 'fineturn_data.npy')
        self.pos_datas = []
        self.neg_datas = []

        self._batch_size = self._opt.batch_size
        self._pos_counts = int(self._batch_size / 4)
        self._neg_counts = int(self._batch_size / 4 * 3)

        self._pos_cursor = 0
        self._neg_cursor = 0
        
        self._img_size = self._opt.image_size        
        self._threshold = self._opt.finetune_threshold

        self.epoch = 0
        self.cursor = 0
        
        # read dataset
        if os.path.exists(self.save_path):
            self._load_from_numpy()
        else:
            self._load_dataset()

    def _load_dataset(self):
        file_path = os.path.join(self._root, self._opt.finetune_list)
        with codecs.open(file_path, 'r', 'utf-8') as fr:
            lines = fr.readlines()
            for ind, line in enumerate(lines):
                context = line.strip().split()
                image_path = os.path.join(self._root, context[0])
                index = int(context[1])
                ref_rect = context[2].split(',')
                ground_truth = [int(i) for i in ref_rect]

                images, vertices, _ = image_proposal(image_path, self._img_size)
                for img_float, proposal_vertice in zip(images, vertices):
                    iou_val = IOU(ground_truth, proposal_vertice)
                    if iou_val < self._threshold:
                        label = 0
                        self.neg_datas.append([img_float, label])
                    else:
                        label = index
                        self.pos_datas.append([img_float, label])
                    self.datas.append([img_float, label])
        joblib.dump(self.datas, self.save_path)


    def _load_from_numpy(self):        
        self.datas = joblib.load(self.save_path)
        for item in self.datas:
            if item[1] == 0:
                self.neg_datas.append([item[0], item[1]])
            else:
                self.pos_datas.append([item[0], item[1]])

    def get_batch(self):
        images = np.zeros((self._batch_size, self._img_size, self._img_size, 3))
        labels = np.zeros((self._batch_size, 1))
           
        count = 0
        while (count < self._pos_counts):
            images[count] = self.pos_datas[self._pos_cursor][0]
            labels[count] = self.pos_datas[self._pos_cursor][1]
            count += 1
            self._pos_cursor += 1
            if self._pos_cursor >= len(self.pos_datas):
                self._pos_cursor = 0
                np.random.shuffle(self.pos_datas)
        while (count < self._pos_counts + self._neg_counts):
            images[count] = self.neg_datas[self._neg_cursor][0]
            labels[count] = self.neg_datas[self._neg_cursor][1]
            count += 1
            self._neg_cursor += 1
            if self._neg_cursor >= len(self.neg_datas):
                self._neg_cursor = 0
                np.random.shuffle(self.neg_datas)
        """
        state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(state)
        np.random.shuffle(labels)
        """
        return images, labels
        
    
    def __len__(self):
        return len(self.datas)

class SVMDataset(DatasetBase):

    def __init__(self, opt):
        super(SVMDataset, self).__init__(opt)
        self._name = 'SVMDataset'

        self.datas = []
        self.classA_features = []
        self.classA_labels = []
        self.classB_features = []
        self.classB_labels = []
        self.save_path = os.path.join(self._opt.generate_save_path, 'svm_data.npy')

        self._img_size = self._opt.image_size        
        
        self.cursor = 0

        self.model = ModelsFactory.get_by_name('FineModel', self._opt, is_train=False)

        # read dataset
        if os.path.exists(self.save_path):
            self._load_from_numpy()
        else:
            self._load_dataset()

    def _load_dataset(self):
        file_path = os.path.join(self._root, self._opt.finetune_list)
        with codecs.open(file_path, 'r', 'utf-8') as fr:
            lines = fr.readlines()
            for ind, line in enumerate(lines):
                
                context = line.strip().split()
                image_path = os.path.join(self._root, context[0])
                index = int(context[1])
                ref_rect = context[2].split(',')
                ground_truth = [int(i) for i in ref_rect]

                images, vertices, _ = image_proposal(image_path, self._img_size)
                for img_float, proposal_vertice in zip(images, vertices):
                    iou_val = IOU(ground_truth, proposal_vertice)
                    if iou_val < self._opt.svm_threshold:
                        label = 0
                    else:
                        label = index

                    px = float(proposal_vertice[0]) + float(proposal_vertice[4] / 2.0)
                    py = float(proposal_vertice[1]) + float(proposal_vertice[5] / 2.0)
                    ph = float(proposal_vertice[5])
                    pw = float(proposal_vertice[4])

                    gx = float(ref_rect[0])
                    gy = float(ref_rect[1])
                    gw = float(ref_rect[2])
                    gh = float(ref_rect[3])

                    box_label = np.zeros(5)
                    box_label[1:5] = [(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)]
                    if iou_val < self._opt.reg_threshold:
                        box_label[0] = 0
                    else:
                        box_label[0] = 1

                    input_data=torch.Tensor([img_float]).permute(0,3,1,2)
                    
                    feature, final = self.model._forward_test(input_data)
                    feature = feature.data.cpu().numpy()

                    self.datas.append([index, feature, label, box_label])
        joblib.dump(self.datas, self.save_path)

    def _load_from_numpy(self):
        self.datas = joblib.load(self.save_path)
        for item in self.datas:
            if item[0] == 1:
                self.classA_features.append(item[1][0])
                self.classA_labels.append(item[2])
            elif item[0] == 2:
                self.classB_features.append(item[1][0])
                self.classB_labels.append(item[2])

    def get_datas(self):
        return self.classA_features, self.classA_labels, self.classB_features, self.classB_labels

    def __len__(self):
        return len(self.datas)

                    
class RegDataset(DatasetBase):

    def __init__(self, opt):
        super(RegDataset, self).__init__(opt)
        self._name = 'RegDataset'

        self.datas = []
        self._batch_size = self._opt.batch_size
        self.cursor = 0
        
        self._load_dataset()

    def _load_dataset(self):
        file_path = os.path.join(self._opt.generate_save_path, 'svm_data.npy')
        self.datas = joblib.load(file_path)

    def get_batch(self):
        images = np.zeros((self._batch_size, 4096))
        labels = np.zeros((self._batch_size, 5))
        
        count = 0
        while( count < self._batch_size):
            images[count] = self.datas[self.cursor][1]
            labels[count] = self.datas[self.cursor][3]
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.datas):
                self.cursor = 0
                np.random.shuffle(self.datas)
        return images, labels

    def __len__(self):
        return len(self.datas)

        

    

    
