from utils import loss
from utils.config import cfg
from utils.mixup import *
from utils.ramps import exp_rampup
from utils.utils import *
import mindspore.communication.management as distributed
import numpy
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
import time
import logging


# 设置MindSpore的计算环境和设备
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")  # 可根据需要选择设备

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = distributed.get_group_size()
    if world_size < 2:
        return inp
    reduced_inp = inp
    distributed.all_reduce(reduced_inp, "sum")
    reduced_inp /= world_size
    return reduced_inp

def update_ema(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    return ema_model


# 初始化MindSpore运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")  # 可根据需要选择设备

def validate(config, testloader, model, writer_dict, device):
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()

    # chalearn
    g_avg = AverageMeter()

    loss_fn = nn.L1Loss(reduction='mean')

    for i_iter, batch in enumerate(testloader):
        train_mask = ops.zeros(batch[0].size(0)).byte()
        # images, labels, age, name = batch

        # chalearn
        images, labels, age, name, sigma = batch

        images = images.to(device)
        age = age.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        outputs, feat = model(images, labels, train_mask)
        ages = ops.sum(outputs * ops.Tensor([i for i in range(101)]).cuda(), dim=1)
        loss1 = loss_fn(ages, age)
        reduced_loss = reduce_tensor(loss1)
        ave_loss.update(reduced_loss.item())

        # chalearn
        age_epsilon = 1 - numpy.exp(-(numpy.square(ages.float().cpu().clone().detach().numpy()
                                                   - age.float().cpu().clone().detach().numpy())
                                      / (2 * numpy.square(sigma.cpu().clone().detach().numpy()))))
        E_error = numpy.mean(age_epsilon)
        g_avg.update(E_error.item(), len(labels))

    print_loss = ave_loss.average() / world_size

    # chalearn
    print('Gaussian Error:', g_avg.avg)

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_mae', print_loss, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return print_loss


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, extloader, optimizer, model, writer_dict, device, scheduler, ema_model):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    local_rank = get_rank()
    world_size = get_world_size()
    scheduler = scheduler

    for i_iter, batch in enumerate(zip(trainloader, extloader)):
        t_size, e_size = batch[0][0].shape[0], batch[1][0].shape[0]
        train_mask = ops.cat((ops.ones(t_size), ops.zeros(e_size))).byte()
        images, labels = ops.cat((batch[0][0], batch[1][0])), ops.cat((batch[0][1], batch[1][1]))
        age, name = ops.cat((batch[0][2], batch[1][2])), batch[0][3]+batch[1][3]
        images = images.to(device)
        labels = labels.to(device)
        age = age.to(device)
        train_mask = train_mask.to(device)
        model.clear_gradients()
        outputs, feat = model(images, labels, train_mask)  # outputs [lab, ulb, lab]
        ages_1 = ops.sum(outputs[:t_size] * ops.Tensor([i for i in range(101)]).cuda(), dim=1)
        ages_2 = ops.sum(outputs[t_size+e_size:] * ops.Tensor([i for i in range(101)]).cuda(), dim=1)
        a1, a2, b1, b2 = 0.5, 0.5, 0.5, 0.5
        loss1 = loss.kl_loss(outputs[:t_size], labels[:t_size]) * a1 + loss.kl_loss(outputs[t_size + e_size:], labels[:t_size]) * a2
        loss2 = loss.L1_loss(ages_1, age[:t_size]) * b1 + loss.L1_loss(ages_2, age[:t_size]) * b2
        loss3 = loss.OEconLoss(feat, labels[0:t_size], age[0:t_size])
        # update mean-teacher
        ema_model = update_ema(model, ema_model, alpha=0.97, global_step=i_iter + 1)
        with ops.context():
            ema_outputs, _ = ema_model(images, labels, train_mask)  # 存在两个选择，保留标签数据的哪个部分
            ema_outputs = ema_outputs[t_size:][:].detach()

        # mixup_loss
        mixed_x, y_a, y_b, lam = mixup_two_targets(images, ema_outputs, 1.0, device, is_bias=False)
        # age
        # y_a = torch.sum(y_a * torch.Tensor([i for i in range(101)]).cuda(), dim=1)
        # y_b = torch.sum(y_b * torch.Tensor([i for i in range(101)]).cuda(), dim=1)
        mixed_outputs, _ = model(mixed_x, labels, train_mask)
        mix_loss = mixup_ce_loss_soft(mixed_outputs[t_size:][:], y_a, y_b, lam)
        rampup = exp_rampup(30)
        mix_loss *= rampup(epoch)

        # loss3 = 0
        total_loss = loss1 + loss2 + mix_loss + loss3
        # total_loss = loss1 + loss2 + mix_loss + loss3 + ema_loss
        reduced_loss = reduce_tensor(total_loss)
        reduced_loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss
        ave_loss.update(reduced_loss.asnumpy().item())
        if cfg.train.optimizer != "ADAM":
            lr = adjust_learning_rate(optimizer,
                                      base_lr,
                                      num_iters,
                                      i_iter + cur_iters, )
        else:

            scheduler.step(i_iter + cur_iters)
            lr = optimizer.param_groups[0]['lr']
            # print(optimizer.param_groups[
            # dongxin add
            ema_age = ops.sum(ema_outputs[:e_size] * ops.Tensor([i for i in range(101)]).cuda(), dim=1)
            ema_loss = loss.L1_loss(ema_age, age[t_size:])
            if i_iter % config.log.print_freq == 0 and local_rank == 0:
                print_loss = ave_loss.average() / world_size
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.6e}, klloss:{:.3f} mloss:{:.3f} mixloss:{:.3f}, clloss:{:.3f}, ema_loss:{:.3f}'. \
                    format(epoch, num_epoch, i_iter, epoch_iters, batch_time.average(),
                           lr, loss1, loss2, mix_loss, loss3, ema_loss)

                logging.info(msg)
                writer.add_scalar('train_loss', print_loss, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1