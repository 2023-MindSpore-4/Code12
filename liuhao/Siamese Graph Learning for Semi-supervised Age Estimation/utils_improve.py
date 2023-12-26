from utils.config import cfg
import os
import time
import logging
from pathlib import Path
import numpy as np
from mindspore.nn import cosine_decay_lr
import mindspore.communication.management as distributed
from mindspore import nn

def make_optimizer(cfg, model, lr, distributed):
    if distributed:
        ignored_params = list(map(id, model.module.fc.parameters()))
        igno_params = filter(lambda p: id(p) in ignored_params and p.requires_grad, model.module.parameters())
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.module.parameters())
    else:
        ignored_params = list(map(id, model.fc.parameters()))
        igno_params = filter(lambda p: id(p) in ignored_params and p.requires_grad, model.parameters())
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.parameters())

    if cfg.train.optimizer == 'SGD':
        optimizer_function = nn.SGD
        kwargs = {
            'momentum': cfg.train.SGD_params.momentum,
            'dampening': cfg.train.SGD_params.dampening,
            'nesterov': cfg.train.SGD_params.nesterov
        }
    elif cfg.train.optimizer == 'ADAM':
        optimizer_function = nn.Adam
        kwargs = {
            'beta1': cfg.train.ADAM_params.beta1,
            'beta2': cfg.train.ADAM_params.beta2,
            'eps': cfg.train.ADAM_params.epsilon,
            'amsgrad': cfg.train.ADAM_params.amsgrad
        }
    elif cfg.train.optimizer == 'ADAMAX':
        optimizer_function = nn.AdaMax
        kwargs = {
            'beta1': cfg.train.ADAM_params.beta1,
            'beta2': cfg.train.ADAM_params.beta2,
            'eps': cfg.train.ADAM_params.epsilon,
        }
    elif cfg.train.optimizer == 'RMSprop':
        optimizer_function = nn.RMSProp
        kwargs = {
            'epsilon': cfg.train.RMSprop_params.epsilon,
            'momentum': cfg.train.RMSprop_params.momentum,
        }
    else:
        raise Exception()

    kwargs['lr'] = lr
    if cfg.model.pretrained:
        lr_b = lr * cfg.model.finetune_factor
    else:
        lr_b = lr
    kwargs['weight_decay'] = cfg.train.weight_decay

    return optimizer_function([{'params': base_params, 'lr': lr_b}, {'params': igno_params, 'lr': lr}], **kwargs)


def create_logger():
    root_output_dir = Path(cfg.log.output_dir)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.model.dataset_name
    model = cfg.model.model_name

    final_output_dir = root_output_dir / Path(dataset + '_' + model)
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)

    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=final_log_file, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.log.log_dir) / dataset / model / Path('_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    if cfg.model.pretrained:
        optimizer.param_groups[0]['lr'] = lr*cfg.model.finetune_factor
    else:
        optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr  # classfier
    return lr


def get_world_size():
    if not distributed.get_rank():
        return 1
    return distributed.get_group_size()


def get_rank():
    if not distributed.get_group_size():
        return 0
    return distributed.get_rank()


def cosine_distance(x, y):
    if x.ndim == 1:
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
    elif x.ndim == 2:
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)

    np.seterr(divide='ignore', invalid='ignore')
    s = np.dot(x, y.T) / (x_norm * y_norm)
    s *= -1
    dist = s + 1
    dist = np.clip(dist, 0, 2)
    if x is y or y is None:
        dist[np.diag_indices_from(dist)] = 0.0
    if np.any(np.isnan(dist)):
        if x.ndim == 1:
            dist = 1.
        else:
            dist[np.isnan(dist)] = 1.
    return dist


class GradualWarmupScheduler():
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: target learning rate = base lr * multiplier if multiplier > 1.0.
                      if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
          total_epoch: target learning rate is reached at total_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def warmup_scheduler(optimizer, n_iter_per_epoch):
  # 不会重新设置学习率，因为T_max指定epoch 为最后一个
   cosine_scheduler = cosine_decay_lr(
       min_lr=0.000001,
       total_step=(cfg.train.epochs - cfg.train.warmup_epoch) * n_iter_per_epoch)
   scheduler = GradualWarmupScheduler(
       optimizer,
       multiplier=cfg.train.multiplier,
       total_epoch=cfg.train.warmup_epoch * n_iter_per_epoch,
       after_scheduler=cosine_scheduler)
   return scheduler

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
