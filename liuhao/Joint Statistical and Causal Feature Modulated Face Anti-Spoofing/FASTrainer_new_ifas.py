import os
from random import randint
from mindspore.ops import operations as P

from train_code.base import BaseTrainer
from utils.statistic import *
from pandas.core.frame import DataFrame
import pandas as pd
from utils.meters import AvgMeter
from metric_margin import CosineMarginProduct
import time
import logging
from utils.utils_new_ifas import adjust_learning_rate
from utils.eval import add_visualization_to_tensorboard, predict, calc_accuracy
import os
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor
from mindspore.train import Model

class FASTrainer():
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, trainloader, valloader, writer, pretrained, epoch_iters, num_iters):

        self.network = network
        self.pretrained = pretrained
        self.start_epoch = 0
        self.trainloader = trainloader
        self.valloader = valloader
        self.writer = writer
        self.cfg = cfg
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.epoch_iters = epoch_iters
        self.num_iters = num_iters

        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric_2 = AvgMeter(writer=writer, name='Accuracy_2/train', num_iter_per_epoch=len(self.trainloader),
                                         per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric_2 = AvgMeter(writer=writer, name='Accuracy_2/val', num_iter_per_epoch=len(self.valloader))

        # Get gradient function
        self.grad_fn = ops.value_and_grad(self.forward_fn, None, optimizer.parameters, has_aux=True)

    def load_model(self):
        print('self.cfg', self.cfg['output_dir'])
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        print('saved_name', saved_name)

    def test_load_model(self):
        saved_name = '/data/dx/HFM_218_1/exp/hfm_218_1__oulu_p2_cssr.pth'
        state = mindspore.load_checkpoint(saved_name)

        # self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'], strict=False)
        # self.start_epoch = state['epoch']


    def save_model(self, epoch):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
    # Define forward function
    def forward_fn(self,data, depth_map,bce_loss):
        net_depth_map, _, _, _, _, _, output, output_env, _= self.network(data)
        loss = self.criterion(net_depth_map, depth_map) + 0.1 * bce_loss
        return loss,net_depth_map

    # Define function of one-step training
    def train_step(self,data, depth_map,bce_loss):
        (loss, _), grads = self.grad_fn(data, depth_map,bce_loss)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss
    def train_one_epoch(self, epoch):
        # self.network.train()
        self.train_loss_metric.reset(epoch)
        # self.train_acc_metric.reset(epoch)
        self.train_acc_metric_2.reset(epoch)
        cosface_func = CosineMarginProduct()
        cross_entroy = nn.CrossEntropyLoss()
        for i, (img, depth_map, label, _) in enumerate(self.trainloader):
            cur_iters = epoch * self.epoch_iters + i
            squeeze = ops.Squeeze(1)
            img = squeeze(img)
            net_depth_map, _, _, _, _, _, output, output_env, _ = self.network(img)

            output_margin = cosface_func(output, label.long())
            output_softmax = ops.softmax(output_margin, axis=1)
            label = label.astype(mindspore.int32)
            # q2 = label
            bce_loss = cross_entroy(output_softmax, label)
            loss = self.train_step(img, depth_map,bce_loss)
            preds, _ = predict(net_depth_map)
            accuracy_2 = calc_accuracy(preds, label)
            self.train_acc_metric_2.update(accuracy_2)
            # Update metrics
            self.train_loss_metric.update(loss)

            if i % 50 == 0:
                # print('Epoch: {}, iter: {}, loss: {}, acc_2: {}, lr: {}'.format(epoch, i, self.train_loss_metric.avg, self.train_acc_metric_2.avg, '0.1'))
                print('Epoch: {}, iter: {}, loss: {}, acc_2: {}'.format(epoch, i, self.train_loss_metric.avg, self.train_acc_metric_2.avg))

    def train(self):
        if self.pretrained:
            self.load_model()
        # intra domain
        self.min_acer = 1.0
        # thresh fine tune
        # self.best_thresh = 0.0
        # cross domain
        # self.min_hter = 1.0
        for epoch in range(self.start_epoch, self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            # intra domain
            if (epoch + 1) % 5 == 0:
                epoch_acc, acer = self.validate(epoch)
                # thresh fine tune
                # epoch_acc, acer, thresh = self.validate(epoch)

                # intra domain
                if acer < self.min_acer:
                    # thresh fine tune
                    # self.best_thresh = threh
                    self.min_acer = acer
                    print('min_acer', self.min_acer)

            # thresh fine tune
            # epoch_acc, acer = self.test(epoch, self.best_thresh)

            # # cross domain
            # if (epoch + 1) % 5 == 0:
            #     epoch_acc, hter = self.validate(epoch)
            #     print('epoch_acc', epoch_acc)
            #     if hter < self.min_hter:
            #         self.min_hter = hter
            #         print('min_hter', self.min_hter)

            self.save_model(epoch)

    def test(self):
        self.test_load_model()

        # intra domain
        self.min_acer = 1.0
        # thresh fine tune
        # self.best_thresh = 0.0

        # cross domain
        # self.min_hter = 1.0
        acer = self.test_one_epoch()

        # thresh fine tune
        # epoch_acc, acer = self.test(epoch, self.best_thresh)

        # # cross domain
        # if (epoch + 1) % 5 == 0:
        #     epoch_acc, hter = self.validate(epoch)
        #     print('epoch_acc', epoch_acc)
        #     if hter < self.min_hter:
        #         self.min_hter = hter
        #         print('min_hter', self.min_hter)

        # self.save_model(epoch)


    def validate(self, epoch):
        # self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)
        self.val_acc_metric_2.reset(epoch)

        seed = randint(0, len(self.valloader)-1)

        # dongxin add 1213
        pred_list = []

        score_list = []
        label_list = []
        target_list = []
        video_list = []
        ss = self.valloader
        for i, (img, depth_map, label, video) in enumerate(self.valloader):
            x = img
            # .to(self.device)
            # img, depth_map, label = img.cuda(), depth_map.cuda(), label.cuda()
            # 0425 add
            squeeze = ops.Squeeze(1)
            img = squeeze(img)
            net_depth_map, _, _, _, _, _, _, _, _ = self.network(img)
            loss = self.criterion(net_depth_map, depth_map)

            preds, score = predict(net_depth_map)
            targets, _ = predict(depth_map)

            # np.append(score_list, score.cpu().data.numpy())
            # np.append(label_list, label.cpu().data.numpy())
            pred_list.append(preds.data.cpu().numpy())

            score_list.append(score.data.cpu().numpy())
            label_list.append(label.data.cpu().numpy())
            target_list.append(targets.data.cpu().numpy())
            video_list.append(video)
            # np.append(target_list, targets)
            # print('targets', targets.size())
            accuracy = calc_accuracy(preds, targets)
            accuracy_2 = calc_accuracy(preds, label.cpu())

            # Update metrics
            self.val_loss_metric.update(loss.item())
            self.val_acc_metric.update(accuracy)
            self.val_acc_metric_2.update(accuracy_2)

            # if i == seed:
            #     add_visualization_to_tensorboard(self.cfg, epoch, img, preds, targets, score, self.writer)

        pred_list = np.concatenate(pred_list)
        for i in np.unique(pred_list):
            print(str(i) + ':' + str(np.sum(pred_list == i)))

        score_list = np.concatenate(score_list)
        label_list = np.concatenate(label_list)
        target_list = np.concatenate(target_list)
        # print('video_list', video_list )
        video_list = np.concatenate(video_list)
        raw_df_list ={'score': score_list, 'label': target_list, 'video': video_list}
        raw_df = DataFrame(raw_df_list)
        raw_df = raw_df.groupby('video')['score', 'label'].mean()
        # print('raw_df', raw_df)
        score_list_2 = raw_df['score'].values.tolist()
        score_list_2 = np.array(score_list_2)
        target_list_2 = raw_df['label'].values.tolist()
        target_list_2 = np.array(target_list_2)
        # print('score', score_list)
        # print('target_list_2', target_list_2)

        print('self.val_acc_metric_2', self.val_acc_metric_2.avg)

        # intra domain
        cur_EER_valid, threshold, _, _ = get_EER_states(score_list, target_list)
        APCER, BPCER, ACER, ACC = calculate(score_list, target_list, threshold)
        cur_EER_valid2, threshold2, _, _ = get_EER_states(score_list_2, target_list_2)
        APCER2, BPCER2, ACER2, ACC2 = calculate(score_list_2, target_list_2, threshold2)
        print('APCER, BPCER, ACER, ACC, threshold', APCER, BPCER, ACER, ACC, threshold)
        print('APCER2, BPCER2, ACER2, ACC2, threshold2', APCER2, BPCER2, ACER2, ACC2, threshold2)

        # # cross domain
        # cur_EER_valid2, threshold2, _, _ = get_EER_states(score_list_2, target_list_2)
        # cur_HTER_valid = get_HTER_at_thr(score_list_2, target_list_2, threshold2)
        # auc_score = roc_auc_score(target_list_2, score_list_2)
        # print('cur_HTER_valid, threshold, auc_score', cur_HTER_valid, threshold2, auc_score)

        # intra domain
        return self.val_acc_metric.avg, ACER2

        # cross domain
        # return self.val_acc_metric.avg, cur_HTER_valid

    def test_one_epoch(self):
        self.network.eval()

        seed = randint(0, len(self.valloader)-1)

        pred_list = []

        score_list = []
        label_list = []
        target_list = []
        video_list = []
        with ops.no_grad():
            for i, (img, depth_map, label, video) in enumerate(self.valloader):
                # .to(self.device)
                img, depth_map, label = img.cuda(), depth_map.cuda(), label.cuda()
                # 0425 add
                squeeze = ops.Squeeze(1)
                img = squeeze(img)
                net_depth_map, _, _, _, _, _, _, _, _ = self.network(img)
                loss = self.criterion(net_depth_map, depth_map)

                preds, score = predict(net_depth_map)
                targets, _ = predict(depth_map)

                # np.append(score_list, score.cpu().data.numpy())
                # np.append(label_list, label.cpu().data.numpy())
                pred_list.append(preds.data.cpu().numpy())

                score_list.append(score.data.cpu().numpy())
                label_list.append(label.data.cpu().numpy())
                target_list.append(targets.data.cpu().numpy())
                video_list.append(video)
                # np.append(target_list, targets)
                # print('targets', targets.size())
                accuracy = calc_accuracy(preds, targets)
                accuracy_2 = calc_accuracy(preds, label.cpu())

                # if i == seed:
                #     add_visualization_to_tensorboard(self.cfg, epoch, img, preds, targets, score, self.writer)

            pred_list = np.concatenate(pred_list)
            for i in np.unique(pred_list):
                print(str(i) + ':' + str(np.sum(pred_list == i)))

            score_list = np.concatenate(score_list)
            label_list = np.concatenate(label_list)
            target_list = np.concatenate(target_list)
            # print('video_list', video_list )
            video_list = np.concatenate(video_list)
            raw_df_list ={'score': score_list, 'label': target_list, 'video': video_list}
            raw_df = DataFrame(raw_df_list)
            raw_df = raw_df.groupby('video')['score', 'label'].mean()
            # print('raw_df', raw_df)
            score_list_2 = raw_df['score'].values.tolist()
            score_list_2 = np.array(score_list_2)
            target_list_2 = raw_df['label'].values.tolist()
            target_list_2 = np.array(target_list_2)
            # print('score', score_list)
            # print('target_list_2', target_list_2)
            # intra domain
            cur_EER_valid, threshold, _, _ = get_EER_states(score_list, target_list)
            APCER, BPCER, ACER, ACC = calculate(score_list, target_list, threshold)
            cur_EER_valid2, threshold2, _, _ = get_EER_states(score_list_2, target_list_2)
            APCER2, BPCER2, ACER2, ACC2 = calculate(score_list_2, target_list_2, threshold2)
            print('APCER2, BPCER2, ACER2, ACC2, threshold2', APCER2, BPCER2, ACER2, ACC2, threshold2)
            # # cross domain
            # cur_EER_valid2, threshold2, _, _ = get_EER_states(score_list_2, target_list_2)
            # cur_HTER_valid = get_HTER_at_thr(score_list_2, target_list_2, threshold2)
            # auc_score = roc_auc_score(target_list_2, score_list_2)
            # print('cur_HTER_valid, threshold, auc_score', cur_HTER_valid, threshold2, auc_score)

            # intra domain
            return ACER2

            # cross domain
            # return self.val_acc_metric.avg, cur_HTER_valid
