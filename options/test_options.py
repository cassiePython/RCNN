from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
       
        self._parser.add_argument('--img_path', type=str, default='./sample_dataset/2flowers/jpg/1/image_1281.jpg', help='file containing results') 
        
        self.is_train = False
