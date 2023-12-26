import os
import time
from datetime import datetime
from os.path import join

import numpy as np
from PIL import Image

from models import EMA
from utils.util import exists, ms_save


class Trainer(object):
    def __init__(
            self,
            args,
            diffusion_model,

            *,
            ema_decay=0.995,
            num_frames=16,
            train_batch_size=32,
            train_lr=1e-4,
            weight_decay=0,
            beta1=0.9,
            beta2=0.999,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            output_dir='./results',
            num_sample_rows=4,
            max_grad_norm=None,
            log=None,
            n_plot=1000,
            n_ckpt=100,
            n_stats=1,
            rank=0,
            root_pkl='model.pth',
            max_ckpt=100,
    ):
        super().__init__()
        self.dl = None
        self.args = args
        self.n_plot = n_plot
        self.n_ckpt = n_ckpt
        self.n_stats = n_stats
        self.rank = rank
        self.root_pkl = root_pkl
        self.max_ckpt = max_ckpt
        self.save_mode = args.save_mode

        self.output_dir = output_dir
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = self.model
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.log = log
        self.step = 0

        self.amp = args.amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows

    def sampler(self, size=None, image_size=None, ckpt_path=-1, ema=True, out_path=None):
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        if not exists(size):
            size = self.batch_size

        self.ema_model.set_train(False)
        self.ema_model.args['test'] = False
        self.ema_model.sample(batch_size=size, use_tqdm=True, image_size=image_size)
        self.ema_model.args['test'] = True
        one_gif = self.ema_model.sample(batch_size=size, use_tqdm=True, image_size=image_size)
        output = []
        for i in range(len(one_gif)):
            one_gif[i] -= one_gif[i].min()
            one_gif[i] /= one_gif[i].max()
            output.append(
                np.concatenate(one_gif[i].transpose(1, 2, 3, 0).mul(255).add(0.5).clamp(0, 255).asnumpy(), axis=1))
        one_gif = np.concatenate(output, axis=0)
        im = Image.fromarray(np.uint8(one_gif))
        im.save('sample.png')

    def test(self, args, root_pkl):
        for _ in range(1):
            out_path = join(args.output, args.name, 'sample')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out_path = join(out_path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            self.sampler(args.test_batch_size, args.image_size, root_pkl, out_path=out_path)
            time.sleep(2)
