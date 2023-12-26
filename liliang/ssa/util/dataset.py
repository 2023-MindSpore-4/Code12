import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import os
from PIL import Image

from mindspore import Tensor

class MyDatasetLoader():

    def __init__(self, dataset_path: list):
        super().__init__()
        self.dataset_path = dataset_path
        self.img = []
        for i in self.dataset_path:
            names = os.listdir(i)
            paths = [i+j for j in names]
            self.img.extend(paths)
        self._index = 0

    def __getitem__(self, index):
        path = self.img[index]
        pic = Image.open(path)
        pic:Tensor = vision.ToTensor()(pic)
        pic = pic.transpose(1,2,0)
        pic = vision.Resize([256, 256])(pic)
        pic = pic.transpose(2,0,1)

        return pic

    def __next__(self):
        if self._index >= self.__len__():
            raise StopIteration
        else:
            pic = self.__getitem__(self._index)
            self._index += 1
            return pic

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.img)
