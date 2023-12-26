# -*- coding: utf-8 -*
import os
# cross_datasets
from mindspore.communication import get_group_size

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'

import pprint

from mindspore import SummaryCollector


import mindspore
from mindspore import nn
from mindspore import context
import mindspore.dataset as ds
from mindspore.communication import init
from mindspore.dataset import DistributedSampler
from utils import data

from model.resnet_gen import resnet50, resnet34, resnet18


# import mindspore.ops as ops
import mindspore.communication as com


from function.function_all import train, validate
import argparse
import ssl
from utils.config import cfg
from utils.utils import *



ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser(description='Age Estimator')
    parser.add_argument('--config', type=str, default='/data/dx/sgl/lsm_debug/non_local/configs/morph/LSM.yml',
                        help='config file path')
    parser.add_argument("--local_rank", type=int, default=0)  
    args, unknown = parser.parse_known_args()
    cfg.merge_from_file(args.config)  # merge config file args
    cfg.merge_from_list(unknown)  # merge command args
    cfg.freeze()
    return args


def init_model(cfg):
    models = {'Resnet50': resnet50, 'Resnet50_pre': resnet50,
              'Resnet34_pre': resnet34, 'Resnet34': resnet34, 'Resnet18': resnet18}
    model = cfg.model.model_name
    assert model in models
    if model == 'effnet':
        model = models[model].from_name('efficientnet-b0', num_classes=101)
        
    elif model == 'Resnet50_pre' or model == 'Resnet50':
        model = models[model](pretrained=True)
        fc_in_features = model.fc.in_features
        model.fc = mindspore.nn.Dense(fc_in_features, 101)
    elif model == 'Resnet18':
        model = models[model](pretrained=True)
        
        fc_in_features = model.fc.in_features
        # model.fc = torch.nn.Linear(fc_in_features, 101)
        model.fc = mindspore.nn.Dense(fc_in_features, 101)
    elif model == 'Resnet34_pre' or model == 'Resnet34':
        model = models[model](pretrained=True)
        fc_in_features = model.fc.in_features
        model.fc = mindspore.nn.Dense(fc_in_features, 101)
    else:
        model = models[model]()
    return model


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger()
    summary_collector = SummaryCollector(summary_dir=tb_log_dir)
    writer_dict = {
        'writer': summary_collector,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
    device = context.get_context("device_id")

    # device = torch.device('cuda:1')
    model = init_model(cfg)  # 初始化model
    ema_model = init_model(cfg)

    if distributed:
        model = mindspore.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ema_model = mindspore.nn.SyncBatchNorm.convert_sync_batchnorm(ema_model)
    
    if distributed:
       
        mindspore.set_context(device_id=args.local_rank)
       
        com.init(backend_name="nccl")
        
        init()
       
        device_num = get_group_size()
        mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL, device_num=device_num)
        device = context.get_context("device_id")  # 获取当前的gpu设备id
        model = init_model(cfg)  # 初始化model
        ema_model = init_model(cfg)


    logger.info(pprint.pformat(cfg))

   
    def create_dataset(dataset_dir, batch_size, num_workers, sampler, shuffle):
        data_set = ds.ImageFolderDataset(dataset_dir=dataset_dir,
                                         batch_size=batch_size,
                                         num_parallel_workers=num_workers,
                                         sampler=sampler,
                                         shuffle=True)
        return data_set

    # semi_data prepared
    if distributed:
        train_sampler = DistributedSampler(10, 5)  # 默认shuffle
        ext_sampler = DistributedSampler(10, 5)
        val_sampler = DistributedSampler(10, 5)
        shuffle = None
    else:
        train_sampler = None
        val_sampler = None
        ext_sampler = None
        shuffle = True
    train_dataset = create_dataset(dataset_dir=data.Data(cfg, mode="train").dataset,
                                   batch_size=cfg.train.train_batch_size,
                                   num_workers=cfg.model.nThread,
                                   sampler=train_sampler,
                                   shuffle=shuffle
                                   )
    ext_dataset = create_dataset(dataset_dir=data.Data(cfg, mode="ext").dataset,
                                 batch_size=cfg.train.ext_batch_size,
                                 num_workers=cfg.model.nThread,
                                 sampler=ext_sampler,
                                 shuffle=shuffle
                                 )
    val_dataset = create_dataset(dataset_dir=data.Data(cfg, mode="val").dataset,
                                 batch_size=cfg.train.val_batch_size,
                                 num_workers=cfg.model.nThread,
                                 sampler=val_sampler,
                                 shuffle=None
                                 )
    # optimizer
    optimizer = make_optimizer(cfg, model, cfg.train.lr, distributed)
    gpu_num = get_group_size()
    data_num = data.Data(cfg, 'train').dataset.__len__()
    epoch_iters = np.int_(data_num / cfg.train.train_batch_size / gpu_num)
    min_mae = 20
    last_epoch = 0
    if not distributed:
        model = model
        ema_model = ema_model
    else:
        model = model.module
        ema_model = ema_model.module
    if cfg.model.pretrained:  
        path = cfg.model.pretrained_path
        state_dict = mindspore.load_checkpoint(path)

        
        if path[-3:] == 'tar':
            state_dict = state_dict['state_dict']
       
        mindspore.load_param_into_net(model, state_dict)


    if cfg.model.resume:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            
            checkpoint = mindspore.load_checkpoint(model_state_file)
            min_mae = checkpoint['min_mae']
            last_epoch = checkpoint['epoch']
            '''模型参数的加载'''
            model.load_state_dict(checkpoint['state_dict'])
            if cfg.train.optimizer == 'ADAM':
                optimizer.load_param_into_net(checkpoint['optimizer'])  # if adam resume sgd will make error
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    end_epoch = cfg.train.epochs
    num_iters = cfg.train.epochs * epoch_iters

    # if distributed and torch.distributed.get_rank() == 0:
    #     wandb.watch(model, log='all', log_freq=10)
    # elif not distributed:
    #     wandb.watch(model, log='all', log_freq=10)

    scheduler = warmup_scheduler(optimizer, epoch_iters)
    for epoch in range(last_epoch, end_epoch):
        if get_group_size() > 1:
            
        train(cfg, epoch, cfg.train.epochs, epoch_iters, cfg.train.lr, num_iters, train_dataset, ext_dataset,
              optimizer, model, writer_dict, device, scheduler, ema_model)

        # 验证
        mae = validate(cfg, val_dataset, model, writer_dict, device)
        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + '/checkpoint.ckpt.tar'))
            mindspore.save_checkpoint(model, os.path.join(final_output_dir, 'checkpoint.ckpt.tar'))

            if min_mae >= mae:
                min_mae = mae
                mindspore.save_checkpoint(model, os.path.join(final_output_dir, 'best.ckpt'))
            msg = 'MIN_MAE: {:.3f}, Curr_MAE:{:.3f}'.format(min_mae, mae)
            logging.info(msg)


if __name__ == '__main__':
    main()
