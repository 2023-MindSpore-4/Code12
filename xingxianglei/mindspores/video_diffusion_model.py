import os
from math import pi

import mindspore
from mindspore import nn, ops, Tensor, Parameter
from tqdm import tqdm

from layer import extract
from utils.util import exists, default, unnormalize_img, ms_save, ms_load


class GaussianDiffusion(nn.Cell):
    def __init__(
            self,
            denoise_fn,
            *,
            args=None,
            image_size=64,
            num_frames=10,
            text_use_bert_cls=False,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.9
    ):
        super().__init__()
        self.args = args
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = ops.cumprod(alphas, dim=0)
        alphas_cumprod_prev = ops.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.insert_param_to_cell(name, Parameter(val.float()))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', ops.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', ops.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', ops.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', ops.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', ops.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', ops.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * ops.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * ops.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t, train=False):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # if train:
        #     return posterior_mean
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        if self.args['out_mean'] == 'START_X':
            x_recon = self.denoise_fn.construct_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        elif self.args['out_mean'] == 'NOISE':
            x_recon = self.predict_start_from_noise(x, t=t,
                                                    noise=self.denoise_fn.construct_with_cond_scale(x, t, cond=cond,
                                                                                                    cond_scale=cond_scale))
        else:
            raise ValueError('out_mean model is error')

        # if clip_denoised and self.args['out_mean'] == 'NOISE':
        #     s = 1.
        #     if self.use_dynamic_thres:
        #         s = ops.quantile(
        #             rearrange(x_recon, 'b ... -> b (...)').abs(),
        #             self.dynamic_thres_percentile,
        #             axis=-1
        #         )
        #
        #         s.clamp_(min=1.)
        #         s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
        #
        #     # clip by threshold, depending on whether static or dynamic
        #     x_recon = x_recon.clamp(-s, s) / s
        return self.q_posterior(x_start=x_recon, x_t=x, t=t, train=False)

    # @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond,
                                                                 cond_scale=cond_scale)
        noise = ops.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1., use_tqdm=True, test=False):
        # device = self.betas.device
        b = shape[0]
        img = ops.randn(shape)
        i = 0
        if not self.args['train'] and self.args['test']:
            i = 0
            img = ms_load(f'output/input/{i}_0.ckpt')
            # img = normalize_img(torch.load(f'2.pth')[:1]).to(device)
            # img=normalize_img(torch.load(f'output/UCF-JumpingJack/sample/2023_05_08_21_17_25.pth')[111:112].to(device))
            img = self.p_sample(img, ops.full((1,), i, dtype=mindspore.int32), cond=cond,
                                cond_scale=cond_scale)
            return unnormalize_img(img)
        if use_tqdm:
            step = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                        total=self.num_timesteps)
        else:
            step = reversed(range(0, self.num_timesteps))

        for i in step:
            if i == 0 and (not self.args['train']):
                if not os.path.exists(f'output/input'):
                    os.makedirs(f'output/input')
                ms_save(img, f'output/input/{i}_{0}.ckpt')
            img = self.p_sample(img, ops.full((b,), i, dtype=mindspore.int32), cond=cond,
                                cond_scale=cond_scale)
        # img=img.clamp(-1,1)
        return unnormalize_img(img)

    # @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16, use_tqdm=True, image_size=None, channels=None,
               num_frames=None):
        # device = next(self.denoise_fn.parameters()).device

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size if not exists(image_size) else image_size
        channels = self.channels if not exists(channels) else channels
        num_frames = self.num_frames if not exists(num_frames) else num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond=cond,
                                  cond_scale=cond_scale, use_tqdm=use_tqdm)

    # @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b = x1.shape[0]
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = ops.stack([Tensor(t)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, ops.full((b,), i, dtype=Tensor.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: ops.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = ops.linspace(0, timesteps, steps).float()
    alphas_cumprod = ops.cos(((x / timesteps) + s) / (1 + s) * pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return ops.clip(betas, 0, 0.9999)
