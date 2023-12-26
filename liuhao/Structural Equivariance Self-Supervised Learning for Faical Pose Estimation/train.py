import time
import math
import re
import sys
import os
import argparse
import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.dataset as ds
from datasets import Pose_300W_LP
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
from omegaconf import DictConfig
import hydra
from loss import ContrastiveLoss, ConLoss, KL_div_loss, ConsistencyLoss
from model import resnet50
import datasets
import resource
from rand_tps import RandTPS
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
#context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=80, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=80, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='/data/hliu/wyx/dataset/dataset/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='/data/hliu/wyx/dataset/dataset/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    cfg: DictConfig
    cfg = dict(cfg)
    args_un = DictConfig(cfg)
    args = parse_args()
    # cudnn.enabled = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    context.set_context(device_target='GPU',device_id=args.gpu_id)
    print("---------------------------------------")
    print(f"Arguments received: ")
    print("---------------------------------------")
    for k, v in sorted(args_un.__dict__.items()):
        print(f"{k:25}: {v}")
    print("---------------------------------------")

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'HPENet', int(time.time()), args.batch_size)


    model = HPENet(backbone_name='resnet50',
                 backbone_file='resnet50_224_new.ckpt',
                 pretrained=True,
                 gpu_id=gpu)

    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    pose_dataset = Pose_300W_LP(args.data_dir, args.filename_list, batch_size, shuffle=True, num_workers=4)

    trainset = ds.GeneratorDataset(pose_dataset, shuffle=True)
    trainset = trainset.batch(batch_size, drop_remainder=True)
    trainloader = trainset.create_dict_iterator()

    epoch_iters = len(trainloader)

    crit = nn.MSELoss()

    lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * num_epochs,
                            step_per_epoch=step_size_train, decay_epoch=num_epochs)

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    optimizer = nn.Momentum(params=model.trainable_params(), learning_rate=lr, momentum=0.9)

    # milestones = np.arange(num_epochs)
    # milestones = [10, 20]
    """scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)"""

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (images, gt_mat, _, _) in enumerate(trainloader):

            tps = RandTPS(input_size[1], input_size[0],
                               batch_size=batch_size,
                               sigma=tps_sigma,
                               border_padding=eqv_border_padding,
                               random_mirror=eqv_random_mirror,
                               random_scale=(random_scale_low,
                                             random_scale_high),
                               mode=tps_mode).cuda(gpu)
            interp = nn.Upsample(
                size=(args.input_size[1], args.input_size[0]), mode='bilinear', align_corners=True)

            # Forward pass
            pred_mat,local_feature = model(images)

            images_tps = tps(pred_mat)

            pred_tps =  interp(pred_low_tps)
            pred_d = pred_mat.detach()
            pred_d.requires_grad = False
            # no padding in the prediction space
            pred_tps_org = tps(pred_d, padding_mode='zeros')

            loss_eqv = KL_div_loss(ops.LogSoftmax(axis=1)(pred_tps), ops.Softmax(axis=1)(pred_tps_org))
            loss_eqv_value += 0.1 * loss_eqv.asnumpy()

            centers_tps = utils.batch_get_centers(ops.Softmax(axis=1)(pred_tps)[:, 1:, :, :])
            pred_tps_org_dif = tps(pred, padding_mode='zeros')
            centers_tps_org = utils.batch_get_centers(ops.Softmax(axis=1)(pred_tps_org_dif)[:, 1:, :, :])

            loss_lmeqv = nn.MSELoss(centers_tps, centers_tps_org)
            loss_lmeqv_value += 0.2 * loss_lmeqv.asnumpy()

            # Calc loss
            loss = crit(gt_mat, pred_mat)

            loss_total = loss+ loss_lmeqv +loss_eqv + 0.2*ConsistencyLoss(local_feature) + 0.1*ContrastiveLoss(local_feature)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch + 1,
                          num_epochs,
                          i + 1,
                          len(pose_dataset) // batch_size,
                          loss.item(),
                      )
                      )

        if b_scheduler:
            scheduler.step()

        # Save models at numbered epochs.
        if (epoch) % 10 == 0:
            print('===> Saving model...')
            save_checkpoint(net_m, f'./ckpt/{summary_name}.ckpt')

if __name__ == '__main__':
    main()
