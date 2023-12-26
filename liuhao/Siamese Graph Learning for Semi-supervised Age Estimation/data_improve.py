import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as vision_transforms
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context
from mindspore import dataset

class Data:
    def __init__(self, cfg, mode):
        self.cfg = cfg
        width = cfg.train.width
        height = cfg.train.height
        transform_list = [
            dataset.transforms.c_transforms.RandomChoice(
                [dataset.vision.c_transforms.RandomHorizontalFlip(),
                 dataset.vision.py_transforms.RandomGrayscale(),
                 dataset.vision.c_transforms.RandomRotation(20),
                 dataset.vision.c_transforms.RandomColorAdjust(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                 ]
            ),
            vision_transforms.Resize([height, width]),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        transform_val = [vision_transforms.RandomHorizontalFlip(p=0),
                         vision_transforms.Resize([height, width]),
                         vision_transforms.ToTensor(),
                         vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ]
        transform_ext = [vision_transforms.Resize([height, width]),
                         vision_transforms.ToTensor(),
                         vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ]
        transform = dataset.transforms.c_transforms.Compose(transform_list)
        transform_v = dataset.transforms.c_transforms.Compose(transform_val)
        self.dataset = Dataset(cfg, transform, mode, transform_v, transform_ext)


class Dataset(ds.Dataset):
    def __init__(self, cfg, transform, mode, trans_v, transform_e):
        self.mode = mode
        self.trans_v = trans_v
        self.trans_e = transform_e
        if mode == "train":
            self.root = cfg.dataset.train_img
            self.label_path = cfg.dataset.train_label
        elif mode == "val":
            self.root = cfg.dataset.val_img
            self.label_path = cfg.dataset.val_label
        elif mode == "ext":
            self.root = cfg.dataset.ext_img
            self.label_path = cfg.dataset.ext_label
        self.transform = transform
        images = []
        label = []

        if mode == "val":
            # chalearn
            sigmas = []

        with open(self.label_path, 'r') as f:
            csv_read = csv.reader(f)
            for line in csv_read:
                images.append(line[0])
                label.append(line[1])
                if mode == "val":
                    # chalearn
                    sigmas.append(line[2])
        self.labels = label[1:]
        self.images = images[1:]

        if mode == "val":
            # chalearn
            self.sigmas = sigmas[1:]

# from torchvision.datasets.folder.default_loader
#self.loader = default_loader
        self.loader = vision_transforms.pil_loader

    def __getitem__(self, index):
        name = self.images[index]
        age = self.labels[index]

        if self.mode == "val":
            # chalearn
            sigma = self.sigmas[index]
            sigma = float(sigma)

        img = self.loader(os.path.abspath(os.path.join(self.root, name)))
        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = Tensor(label, dtype=mstype.float32)
        age = int(age)

        if self.transform is not None and (self.mode == 'train' or self.mode == 'ext'):
            img = self.transform(img)
        elif self.trans_v is not None and self.mode == 'val':
            img = self.trans_v(img)

        if self.mode == "train" or self.mode == 'ext':
            return img, label, age, name
        elif self.mode == "val":
            return img, label, age, name, sigma

    def __len__(self):
        return len(self.labels)


def decompose(name, img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img1 = img * 255
    img1 = img1.asnumpy().transpose(1, 2, 0)
    img1 = img1.astype(dtype=np.uint8)
    plt.imshow(img1)
    plt.show()
    return img1


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
