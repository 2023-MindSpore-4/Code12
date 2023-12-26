# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import absolute_import
from __future__ import print_function

import os
import random

import click
import numpy as np
from sklearn.metrics import jaccard_score, precision_score,\
    recall_score, f1_score, pairwise
# import torch
import mindspore as ms
# import torch.backends.cudnn as cudnn
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
from irgan.models.object_localizer import Inception3ObjectLocalizer

# import visdom

# viz = visdom.Visdom(env="IR-GAN My Experiments")  # 创建环境名为Test1
# ok的


loaded_model = None
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class ImageFolderNonGT():
    """
        Code from https://github.com/pytorch/vision/blob/f566fac80e3182a8b3c0219a88ae00ed1b81d7c7/torchvision/datasets/folder.py
        License:
            BSD 3-Clause License

            Copyright (c) Soumith Chintala 2016,
            All rights reserved.

            Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions are met:

            * Redistributions of source code must retain the above copyright notice, this
              list of conditions and the following disclaimer.

            * Redistributions in binary form must reproduce the above copyright notice,
              this list of conditions and the following disclaimer in the documentation
              and/or other materials provided with the distribution.

            * Neither the name of the copyright holder nor the names of its
              contributors may be used to endorse or promote products derived from
              this software without specific prior written permission.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
            AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
            DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
            FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
            DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
            SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
            CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
            OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    def __init__(self, root, num, dataset, transform=None):
        self.root = root
        self.num = str(num)
        self.transform = transform

        if dataset == 'codraw':
            generated_dirs = sorted(
                [int(d.name) for d in os.scandir(self.root) if d.is_dir() and not d.name.endswith('_gt')], reverse=True)
            gt_dirs = sorted([int(d.name[:-3]) for d in os.scandir(self.root) if d.is_dir() and d.name.endswith('_gt')],
                             reverse=True)

            assert len(gt_dirs) == len(generated_dirs)

            gt_files = []
            generated_files = []

            for i in range(len(gt_dirs)):
                gt_files.extend([os.path.join(str(gt_dirs[i]) + '_gt', x)
                                 for x in os.listdir(os.path.join(self.root, str(gt_dirs[i]) + '_gt'))
                                 if x[0] == self.num])
                generated_files.extend([os.path.join(str(generated_dirs[i]), x)
                                        for x in os.listdir(os.path.join(self.root, str(generated_dirs[i])))
                                        if x[0] == self.num])

        elif dataset == 'iclver':
            generated_dirs = sorted(
                [d.name for d in os.scandir(self.root) if d.is_dir() and not d.name.endswith('_gt')])
            gt_dirs = sorted([d.name for d in os.scandir(self.root) if d.is_dir() and d.name.endswith('_gt')])

            assert len(gt_dirs) == len(generated_dirs)

            gt_files = []
            generated_files = []

            for i in range(len(gt_dirs)):
                gt_files.extend([os.path.join(gt_dirs[i], x) for x in os.listdir(os.path.join(self.root, gt_dirs[i]))
                                 if x[0] == self.num])
                generated_files.extend([os.path.join(generated_dirs[i], x)
                                        for x in os.listdir(os.path.join(self.root, generated_dirs[i]))
                                        if x[0] == self.num])

        print('Right Now Iterations:', num)
        print(len(gt_files))
        print(len(generated_files))

        gt_imgs = [x for x in gt_files if has_file_allowed_extension(x, IMG_EXTENSIONS)]
        generated_imgs = [x for x in generated_files if has_file_allowed_extension(x, IMG_EXTENSIONS)]
        self.gt_filenames = gt_imgs
        self.generated_filenames = generated_imgs

        assert len(self.gt_filenames) == len(self.generated_filenames)

        if len(self.gt_filenames) == 0:
            raise(RuntimeError("Found 0 files in folder: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.loader = default_loader

    def __getitem__(self, idx):
        gt_path = self.gt_filenames[idx]
        generated_path = self.generated_filenames[idx]
        gt_sample = self.loader(os.path.join(self.root, gt_path))
        generated_sample = self.loader(os.path.join(self.root, generated_path))
        if transforms is not None:
            generated_sample = self.transform(generated_sample)
            gt_sample = self.transform(gt_sample)

        return generated_sample, gt_sample, generated_path

    def __len__(self):
        return len(self.gt_filenames)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def setup_inception_model(num_classes, pretrained=False):
    if num_classes == 24:
        num_coords = 2
    else:
        num_coords = 3
    model = Inception3ObjectLocalizer(num_objects=num_classes,
                                                            pretrained=pretrained,
                                                            num_coords=num_coords)
    return model


def construct_graph(coords, dataset):
    n = len(coords)
    graph = np.zeros((2, n, n))
    for i in range(n):
        if coords.shape[1] == 2:
            ref_x, ref_y = coords[i]
        else:
            ref_x, _, ref_y = coords[i]
        for j in range(n):
            if i == j:
                query_x, query_y = 0.5, 0.5
            else:
                if coords.shape[1] == 2:
                    query_x, query_y = coords[j]
                else:
                    query_x, _, query_y = coords[j]

            if ref_x > query_x:
                graph[0, i, j] = 1
            elif ref_x < query_x:
                graph[0, i, j] = -1

            if ref_y > query_y:
                graph[1, i, j] = 1
            elif ref_y < query_y:
                graph[1, i, j] = -1

    return graph


def calculate_metrics(target, pred):
    """
    target: (length, label_num)
    pred: (length, label_num)
    """
    right_count = (target*pred).sum(axis=1)
    target_count = target.sum(axis=1)
    pred_count = pred.sum(axis=1)
    precision = right_count/pred_count
    recall = right_count/target_count
    where_are_NaNs = np.isnan(precision)
    precision[where_are_NaNs] = 0

    where_are_NaNs = np.isnan(recall)
    recall[where_are_NaNs] = 0
    f1 = 2/(1/precision+1/recall)
    return precision, recall, f1

def get_graph_similarity(detections, label, locations, gt_locations, dataset):
    """Computes the accuracy of relationships of the intersected
    detections multiplied by recall
    @args:
        detections: (num_objecs)
        label: (num_objects)
    """
    intersection = (detections & label).astype(bool)
    if not np.any(intersection):
        return 0

    locations = locations.data.cpu().numpy()[intersection]
    gt_locations = gt_locations.data.cpu().numpy()[intersection]

    genereated_graph = construct_graph(locations, dataset)
    gt_graph = construct_graph(gt_locations, dataset)

    matches = ms.Tensor((genereated_graph == gt_graph), ms.int32).flatten()
    matches_accuracy = matches.sum() / len(matches)
    recall = (label*detections).sum()/label.sum()

    graph_similarity = recall * matches_accuracy

    return graph_similarity

# Relative Positional Distance
# RPD_value = RPD(locations,gt_locations)
def RPD(coords, coords2):
    n = len(coords)
    for i in range(n):
        if i==0:
            RPD_value = 0
        if coords.shape[1] == 2:
            ref_x1, ref_y1 = coords[i]  # i-clver
            ref_x2, ref_y2 = coords2[i]  # i-clver
        else:
            ref_x1, _, ref_y1 = coords[i]  # Codraw
            ref_x2, _, ref_y2 = coords2[i]  # Codraw

        RPD_value = RPD_value + pow((pow(ref_x1-ref_x2, 2)+pow(ref_y1-ref_y2, 2)),0.5)
        # print('RPD_value_in_for:\r\n', RPD_value)


    # print('RPD_value:\r\n', RPD_value)
    RPD_value = RPD_value/n
    # print('RPD_value/n:\r\n', RPD_value)
    return RPD_value


def get_PRD(detections, label, locations, gt_locations):
    """Computes the accuracy of relationships of the intersected  # 相交，交叉；横穿，贯穿
    detections multiplied by recall
    计算相交关系的精度
    检测和召回成倍增加
    """
    intersection = (detections & label).astype(bool)
    """np.all(np.array)   对矩阵所有元素做与操作，所有为True则返回True
     np.any(np.array)   对矩阵所有元素做或运算，存在True则返回True
     astype: numpy_ndarry下的不同数据类型的转换，转换成布尔型
    """
    if not np.any(intersection):
        return 0

    locations = locations.data.cpu().numpy()[intersection]  # 把tensor转换成numpy的格式
    gt_locations = gt_locations.data.cpu().numpy()[intersection]
    # print('============================================')
    # print('locations:\r\n', locations)
    # print('locations_shape:\r\n', locations.shape)
    # print('gt_locations:\r\n', gt_locations)
    # print('gt_locations_shape:\r\n', gt_locations.shape)

    # genereated_graph = construct_graph(locations, dataset)
    # gt_graph = construct_graph(gt_locations, dataset)

    # matches = (genereated_graph == gt_graph).astype(int).flatten()  # astype是NDarry下的转换
    # matches_accuracy = matches.sum() / len(matches)
    RPDistence_value = RPD(locations,gt_locations)
    recall = recall_score(label, detections, average='samples')
    # print('recall:\r\n', recall)

    RPDistence_value = recall * RPDistence_value
    # print('recall * RPDistence_value:\r\n', RPDistence_value)

    return RPDistence_value


def get_obj_det_acc(dataloader, dataset):
    jss = []
    cs = []
    graph_similarity = []

    gt_all = []
    pred_all = []
    names = []

    RPD_value = []

    for _, (sample, gt, name) in enumerate(tqdm(dataloader)):
        sample = sample.cuda()
        gt = gt.cuda()
        detection_logits, locations = loaded_model(sample)
        gt_detection_logits, gt_locations = loaded_model(gt)

        pred = detection_logits > 0.5
        gt_pred = gt_detection_logits > 0.5

        pred = pred.cpu().numpy().astype('int')
        gt_pred = gt_pred.cpu().numpy().astype('int')
        gt_detection_logits = gt_detection_logits.cpu().numpy()
        detection_logits = detection_logits.cpu().numpy()

        gt_all.extend(gt_pred)
        pred_all.extend(pred)
        names.extend(name)

        for i in range(sample.shape[0]):
            graph_similarity.append(get_graph_similarity(pred[i], gt_pred[i], locations[i], gt_locations[i], dataset))

        cs.append(pairwise.cosine_similarity(gt_detection_logits, detection_logits)[0][0])
        jss.append(jaccard_score(gt_pred, pred))
        RPD_value.append(get_PRD(pred, gt_pred, locations, gt_locations))

    ps, rs, f1 = calculate_metrics(np.array(gt_all), np.array(pred_all))
    results = pd.DataFrame({
        'p': ps, 'r': rs, 'f1': f1, 'gs': graph_similarity
    }, index=names)
    # results.to_json('results_metrics.json')
    F1_Myself = 2 / ((1 / ps.mean()) + (1 / rs.mean()))

    return np.mean(jss), ps.mean(), rs.mean(), f1.mean(), np.mean(cs), np.mean(graph_similarity), np.mean(RPD_value), F1_Myself


def _init_inception(model_dir):
    global loaded_model

    checkpoint = ms.load_checkpoint(model_dir)
    random.seed(1234)
    ms.set_seed(1234)
    # if checkpoint['cuda_enabled']:
    #     cudnn.deterministic = True
    # loaded_model = setup_inception_model(checkpoint['num_classes'], pretrained=checkpoint['pretrained'])
    # if checkpoint['cuda_enabled']:
    #     loaded_model = loaded_model.cuda()
    #     cudnn.benchmark = True
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.eval()
    return loaded_model


def calculate_inception_objects_accuracy(num, dataset, img_dir, model_path):
    numss = num

    if loaded_model is None:
        _init_inception(model_path)
    test_transforms = transforms.Compose([transforms.Resize(299),
                                          transforms.ToTensor()])
    dataset = ImageFolderNonGT(img_dir, numss, dataset, transform=test_transforms)
    dataset = dataset.batch(batch_size=1)
    with ms.ops.stop_gradient(True):
        jss, avg_precision, avg_recall, avg_f1, cs, graph_similarity, RPDistence, F1_Myself  = get_obj_det_acc(dataloader, dataset)
        print('\nNumber of images used: {}\nJSS: {}\n AP: {}\nAR: {}\n F1: {}\nCS: {}\nGS: {}\nRPD: {}\nF1_Myself: {}'.format(len(dataset), jss,
                                                                                                      avg_precision, avg_recall,
                                                                                                      avg_f1, cs,
                                                                                                      graph_similarity,RPDistence, F1_Myself))
    return jss, avg_precision, avg_recall, avg_f1, cs, graph_similarity, RPDistence, F1_Myself




@click.command()
@click.option('--iterations', default=5, type=int)
@click.option('--dataset', type=str)
@click.option('--img-dir', type=click.Path(exists=True, dir_okay=True, readable=True), required=True)
def test_all_metric(iterations, dataset, img_dir):
    nums = iterations
    print('<1> The iterations you choose is:', nums)  # 1
    if dataset=='codraw':
        print('<2> The dataset you choose is:', dataset)  # 2
        model_path = '/home/ubuntu/conggaoxiang/GeNeVA_datasets/data/codraw_inception_best_checkpoint.pth'
    if dataset=='iclver':
        print('<2> The dataset you choose is:', dataset)
        model_path = '/home/ubuntu/conggaoxiang/GeNeVA_datasets/data/iclevr_inception_best_checkpoint.pth'

    print('<3> Your test file path is:', img_dir)  # 3
    print('<4> model_path:', model_path)  # 4

    jss_all = []
    avg_precision_all = []
    avg_recall_all = []
    avg_f1_all = []
    cs_all = []
    graph_similarity_all = []
    RPDistence_all = []
    F1_Myself_all = []

    for num in range(nums):
        jss_i, avg_precision_i, avg_recall_i, avg_f1_i, cs_i, graph_similarity_i, RPDistence_i, F1_Myself_i = \
            calculate_inception_objects_accuracy(num, dataset, img_dir, model_path)
        jss_all.append(jss_i)
        avg_precision_all.append(avg_precision_i)
        avg_recall_all.append(avg_recall_i)
        avg_f1_all.append(avg_f1_i)
        cs_all.append(cs_i)
        graph_similarity_all.append(graph_similarity_i)
        RPDistence_all.append(RPDistence_i)
        F1_Myself_all.append(F1_Myself_i)

    print('###############################################')
    print('jss_all:', jss_all, '\r\n')
    print('avg_precision_all:', avg_precision_all, '\r\n')
    print('avg_recall_all:', avg_recall_all, '\r\n')
    print('avg_f1_all:', avg_f1_all, '\r\n')
    print('cs_all:', cs_all, '\r\n')
    print('graph_similarity_all:', graph_similarity_all, '\r\n')
    print('RPDistence_all:', RPDistence_all, '\r\n')
    print('F1_Myself_all:', F1_Myself_all, '\r\n')

    # # visdom绘图
    # if dataset=='codraw':
    #     viz.line(
    #         X=list(range(1, 1 + len(avg_f1_all))),  # x坐标
    #         Y=avg_f1_all,  # y值
    #         win="line1",  # 窗口id
    #         name="avg_f1_all",  # 线条名称
    #         update=None,  # 已添加方式加入
    #         opts={
    #             'showlegend': True,  # 显示网格
    #             'title': "CoDraw: F1, Persion and Recall",
    #             'xlabel': "iterations",  # x轴标签
    #             'ylabel': "metric",  # y轴标签
    #         }
    #     )
    #
    #     viz.line(
    #         X=list(range(1, 1 + len(avg_f1_all))),  # x坐标
    #         Y=avg_precision_all,  # y值
    #         win="line1",  # 窗口id
    #         name="avg_precision_all",  # 线条名称
    #         update='append',
    #     )
    #
    #     viz.line(
    #         X=list(range(1, 1 + len(avg_f1_all))),  # x坐标
    #         Y=avg_recall_all,  # y值
    #         win="line1",  # 窗口id
    #         name="avg_recall_all",  # 线条名称
    #         update='append',
    #     )
    #
    #     viz.line(
    #         X=list(range(1, 1 + len(graph_similarity_all))),  # x坐标
    #         Y=graph_similarity_all,  # y值
    #         win="win",  # 窗口id
    #         name="graph_similarity_all",  # 线条名称
    #         update='None',  # 已添加方式加入
    #         opts={
    #             'showlegend': True,  # 显示网格
    #             'title': "CoDraw: graph_similarity_all",
    #             'xlabel': "iterations",  # x轴标签
    #             'ylabel': "metric",  # y轴标签
    #         }
    #     )
    # if dataset=='iclver':
    #     viz.line(
    #         X=list(range(1, 1 + len(avg_f1_all))),  # x坐标
    #         Y=avg_f1_all,  # y值
    #         win="line1",  # 窗口id
    #         name="avg_f1_all",  # 线条名称
    #         update=None,  # 已添加方式加入
    #         opts={
    #             'showlegend': True,  # 显示网格
    #             'title': "iClver: F1, Persion and Recall",
    #             'xlabel': "iterations",  # x轴标签
    #             'ylabel': "metric",  # y轴标签
    #         }
    #     )
    #
    #     viz.line(
    #         X=list(range(1, 1 + len(avg_f1_all))),  # x坐标
    #         Y=avg_precision_all,  # y值
    #         win="line1",  # 窗口id
    #         name="avg_precision_all",  # 线条名称
    #         update='append',
    #     )
    #
    #     viz.line(
    #         X=list(range(1, 1 + len(avg_f1_all))),  # x坐标
    #         Y=avg_recall_all,  # y值
    #         win="line1",  # 窗口id
    #         name="avg_recall_all",  # 线条名称
    #         update='append',
    #     )
    #
    #     viz.line(
    #         X=list(range(1, 1 + len(graph_similarity_all))),  # x坐标
    #         Y=graph_similarity_all,  # y值
    #         win="win",  # 窗口id
    #         name="graph_similarity_all",  # 线条名称
    #         update='None',  # 已添加方式加入
    #         opts={
    #             'showlegend': True,  # 显示网格
    #             'title': "iClver: graph_similarity_all",
    #             'xlabel': "iterations",  # x轴标签
    #             'ylabel': "metric",  # y轴标签
    #         }
    #     )



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5'
    test_all_metric()
