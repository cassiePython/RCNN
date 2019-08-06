from __future__ import division
from models.model_factory import ModelsFactory
from options.test_options import TestOptions
from utils.util import image_proposal
from utils.util import show_rect
import torch
import numpy as np

class Test:
    def __init__(self):
        self._opt = TestOptions().parse()
        self._img_path = self._opt.img_path
        self._img_size = self._opt.image_size

        self.fine_model = ModelsFactory.get_by_name('FineModel', self._opt, is_train=False)
        self.svm_model_A = ModelsFactory.get_by_name('SvmModel', self._opt, is_train=False)
        self.svm_model_A.load('A')
        self.svm_model_B = ModelsFactory.get_by_name('SvmModel', self._opt, is_train=False)
        self.svm_model_B.load('B')
        self.svms = [self.svm_model_A, self.svm_model_B]
        self.reg_model = ModelsFactory.get_by_name('RegModel', self._opt, is_train=False)

        self.test()

    def test(self):
        imgs, _, rects = image_proposal(self._img_path, self._img_size)

        show_rect(self._img_path, rects, ' ')

        input_data=torch.Tensor(imgs).permute(0,3,1,2)
        features, _ = self.fine_model._forward_test(input_data)
        features = features.data.cpu().numpy()

        results = []
        results_old = []
        results_label = []
        count = 0

        flower = {1:'pancy', 2:'Tulip'}

        for f in features:
            for svm in self.svms:
                pred = svm.predict([f.tolist()])
                # not background
                if pred[0] != 0:
                    results_old.append(rects[count])
                    input_data=torch.Tensor(f)
                    box = self.reg_model._forward_test(input_data)
                    box = box.data.cpu().numpy()
                    if box[0] > 0.3:
                        px, py, pw, ph = rects[count][0], rects[count][1], rects[count][2], rects[count][3]
                        old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0
                        x_ping, y_ping, w_suo, h_suo = box[1], box[2], box[3], box[4],
                        new__center_x = x_ping * pw + old_center_x
                        new__center_y = y_ping * ph + old_center_y
                        new_w = pw * np.exp(w_suo)
                        new_h = ph * np.exp(h_suo)
                        new_verts = [new__center_x, new__center_y, new_w, new_h]
                        results.append(new_verts)
                        results_label.append(pred[0])
            count += 1

        average_center_x, average_center_y, average_w,average_h = 0, 0, 0, 0
        #use average values to represent the final result
        for vert in results:
            average_center_x += vert[0]
            average_center_y += vert[1]
            average_w += vert[2]
            average_h += vert[3]
        average_center_x = average_center_x / len(results)
        average_center_y = average_center_y / len(results)
        average_w = average_w / len(results)
        average_h = average_h / len(results)
        average_result = [[average_center_x, average_center_y, average_w, average_h]]
        result_label = max(results_label, key=results_label.count)
        show_rect(self._img_path, results_old, ' ')
        show_rect(self._img_path, average_result, flower[result_label])
      

if __name__ == "__main__":
    Test()
        

            
            
            
        
