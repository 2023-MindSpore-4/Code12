import argparse
import datetime
import os
import shutil

import imageio
import matplotlib
import mindspore
import numpy as np
from skimage.util import img_as_ubyte

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_output_dir(name, output='output', sub=None):
    if sub is None:
        sub = ['samples', 'ckpt', 'part']
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('{}/{}'.format(output, name), t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        for i in sub:
            os.makedirs(os.path.join(output_dir, i))
    return output_dir


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def copy_source(file=None, output_dir='output'):
    if file is None:
        file = []
    for i in file:
        shutil.copyfile(i, os.path.join(output_dir, os.path.basename(i)))


# def common_init(rank, seed, save_dir):
#     # we use different seeds per gpu. But we sync the weights after model initialization.
#     torch.manual_seed(rank + seed)
#     np.random.seed(rank + seed)
#     torch.cuda.manual_seed(rank + seed)
#     torch.cuda.manual_seed_all(rank + seed)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
#     logging = Logger(rank, save_dir)
#
#     return logging


def restore_parts(state_dict_target, state_dict_from):
    # def cy():
    #     for name, param in state_dict_target.items():
    #         yield name, param
    # n = cy()
    for name, param in state_dict_from.items():
        # name2, param2 = next(n)
        # while param2.size() != param.size():
        #     name2, param2 = next(n)
        # state_dict_target[name2].copy_(param)
        # continue
        if name not in state_dict_target:
            print(name)
            continue

        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target


def plot_stats(output_dir, stats, interval):
    content = stats.keys()
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        if len(content) == 1:
            axs.plot(interval, v)
            axs.grid()
            axs.set_ylabel(k)
        else:
            axs[j].plot(interval, v)
            axs[j].grid()
            axs[j].set_ylabel(k)
    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, 'stat.png'), bbox_inches='tight')
    plt.close(f)


def make_gif(images, filename):
    frames_num = images.size(0)
    x = images.permute(1, 2, 3, 0)
    x = x.cpu().detach().numpy()
    frames = []
    for i in range(frames_num):
        frames += [img_as_ubyte(x[i])]
    imageio.mimsave(filename, frames, 'GIF', duration=1)
    return np.array(frames)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def is_odd(n):
    return (n % 2) == 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def normalize_img(t):
    return t * 2 - 1


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def cycle(dl, sam=None):
    i = 0
    while True:
        if exists(sam):
            sam.set_epoch(i)
        i += 1
        for data in dl:
            yield data

def cycle_withsam(dl, sam):
    i = 0
    while True:
        sam.set_epoch(i)
        i += 1
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
#     images = map(T.ToPILImage(), tensor.unbind(dim=1))
#     first_img, *rest_imgs = images
#     first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
#     return images


def noop(*args, **kwargs):
    pass


def get_nrow(n):
    if n <= 1:
        return 1
    else:
        return int(np.sqrt(n - 1) + 1)


def get_time_name():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def get_indi(indices, n, w, dim=1):
    if dim == -1:
        return indices[..., n:n + w]
    elif dim == 0:
        return indices[n:n + w]
    elif dim == 1:
        return indices[:, n:n + w]
    elif dim == 2:
        return indices[:, :, n:n + w]
    elif dim == 3:
        return indices[:, :, :, n:n + w]
    elif dim == 4:
        return indices[:, :, :, :, n:n + w]


def make_gif(images, filename):
    frames_num = images.size(0)
    x = images.permute(1, 2, 3, 0)
    x = x.cpu().detach().numpy()
    frames = []
    for i in range(frames_num):
        frames += [img_as_ubyte(x[i])]
    imageio.mimsave(filename, frames, 'GIF', duration=0.5)
    return np.array(frames)

def ms_save(file,name):
    new_params_list = [{"name": 'file', "data": file}]
    mindspore.save_checkpoint(new_params_list, ckpt_file_name=name)

def ms_load(name):
    return mindspore.load_checkpoint(name)['file']