import numpy as np
from mindspore.dataset import vision
from mindspore.dataset import transforms
import os
from PIL import Image


class PartSHHQDataset:
    def __init__(self, dataset_path, mode='train', img_size=(128, 64)):
        # super().__init__()
        self.img_size = img_size
        self.imgs_path = os.path.join(dataset_path, mode, 'imgs')
        self.labels_path = os.path.join(dataset_path, mode, 'labels')
        self.img_list = os.listdir(self.imgs_path)
        self.img_transforms = transforms.Compose([
            vision.Resize(size=self.img_size),
            vision.ToTensor(),
        ])
        self.label_transforms = transforms.Compose([
            vision.Resize(size=self.img_size),
            vision.Grayscale(num_output_channels=1),
            vision.ToTensor(),
        ])

    def process_label(self, label_path):
        labels_path_list = os.listdir(label_path)
        labels_list = []
        for l_path in labels_path_list:
            label_img = Image.open(os.path.join(label_path, l_path))
            label_img = label_img.resize((self.img_size[1], self.img_size[0]), Image.ANTIALIAS)
            label = np.array(label_img) / 255.
            label = label[:, :, 2:3] * 0.114 + label[:, :, 1:2] * 0.587 + label[:, :, 0:1] * 0.299
            label = np.transpose(label, (2, 0, 1))
            labels_list.append(label)
        return np.concatenate(labels_list, axis=0)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.imgs_path, self.img_list[index])
        pil_img = Image.open(img_path)
        pil_img = pil_img.resize((self.img_size[1], self.img_size[0]), Image.ANTIALIAS)
        img = np.array(pil_img) / 255.
        img = img * 2. - 1.
        img = np.transpose(img, (2, 0, 1))
        img_name = self.img_list[index].split(".")[0]
        label_path = os.path.join(self.labels_path, img_name)
        label = self.process_label(label_path)
        return img.astype(np.float32), label.astype(np.float32)


if __name__ == "__main__":
    from mindspore.dataset import GeneratorDataset
    img_transforms = transforms.Compose([
        vision.Resize(size=(128, 64)),
        vision.ToTensor(),
        vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    label_transforms = transforms.Compose([
        vision.Resize(size=(128, 64)),
        vision.Grayscale(num_output_channels=1),
        vision.ToTensor(),
    ])
    dataset = PartSHHQDataset('/media/lz/新加卷1/datasets/PartSSHQ', img_size=(128, 64))
    dataset = GeneratorDataset(dataset, column_names=['img', 'label'], shuffle=True)
    dataset = dataset.map(operations=img_transforms, input_columns="img")
    dataset = dataset.map(operations=label_transforms, input_columns="label")
    dataset = dataset.batch(batch_size=8)
    train_loader = dataset.create_tuple_iterator()
    for i, x, in enumerate(train_loader):
        mask, template = x
        print(mask.shape)
        print(template.shape)
        break
