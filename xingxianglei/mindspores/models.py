import os
from functools import partial

import mindspore.nn as nn
from mindspore import Tensor, ops, Parameter

from layer import Attention, RelativePositionBias, PreNorm, Residual, SinusoidalPosEmb, ResnetBlock, \
    SpatialLinearAttention, Downsample, Upsample, prob_mask_like, top_ch, top_txy
from rotary import RotaryEmbedding
from utils.util import default, is_odd, exists, ms_save, ms_load


class Unet3D(nn.Cell):
    def __init__(
            self,
            args,
            dim=64,
            cond_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            attn_heads=8,
            attn_dim_head=32,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            block_type='resnet',
            resnet_groups=8,
            use_top=(False, True, False, False),
            top_rate=(0.1, 0.2, 0.3, 0.4, 0.5),
            with_t=(False, True, False, False),
            top_mode=('ch', 'ch', 'op', 'op'),
            top_dim=(1, 2, -1, -1),

    ):
        super().__init__()
        self.block_type = block_type
        self.channels = channels
        self.args = args
        self.amp = args.amp

        # temporal attention and its relative positional encoding

        # rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        class Einops(nn.Cell):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def construct(self, x, **kwargs):
                shape = x.shape
                x = x.view(shape[0], shape[1], shape[2], -1)
                x = ops.permute(x, (0, 3, 2, 1))
                x = self.fn(x, **kwargs)
                x = ops.permute(x, (0, 3, 2, 1))
                x = x.view(*shape)
                return x

        # temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
        #                                             Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
        #                                                       rotary_emb=rotary_emb))
        temporal_attn = lambda dim: Einops(Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
                                                     rotary_emb=RotaryEmbedding(min(32, attn_dim_head))))

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads,
                                                      max_distance=64)  # realistically will not be able to generate
        # that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), pad_mode='pad',
                                   padding=(0, 0, init_padding, init_padding, init_padding, init_padding),
                                   has_bias=True)

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.SequentialCell(
            SinusoidalPosEmb(dim),
            nn.Dense(dim, time_dim),
            nn.GELU(),
            nn.Dense(time_dim, time_dim)
        )
        # text conditioning
        self.has_cond = exists(cond_dim)
        cond_dim = cond_dim
        self.null_cond_emb = Parameter(Tensor.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.CellList([])
        self.ups = nn.CellList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.CellList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads)))
                if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        class Einops2(nn.Cell):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def construct(self, x, **kwargs):
                shape = x.shape
                x = x.view(shape[0], shape[1], shape[2], -1)
                x = ops.permute(x, (0, 2, 3, 1))
                x = self.fn(x, **kwargs)
                x = ops.permute(x, (0, 3, 1, 2))
                x = x.view(*shape)
                return x

        # spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
        spatial_attn = Einops2(Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        bias = False
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            tmp = nn.CellList([
                block_klass_cond(dim_out * ((ind == 0) + 1), dim_in, bias=bias),
                block_klass_cond(dim_in, dim_in, bias=bias),
                Residual(SpatialLinearAttention(dim_in, heads=attn_heads, bias=bias))
                if use_sparse_linear_attn else nn.Identity(),
                Residual(temporal_attn(dim_in)),
                Upsample(dim_in, bias=bias) if not is_last else nn.Identity(),
            ])
            if use_top[ind]:
                if top_mode[ind] == 'ch':
                    tmp.append(top_ch(rate=top_rate[ind], dim=top_dim[ind], act=ops.relu))
                elif top_mode[ind] == 'op':
                    tmp.append(top_txy(rate=top_rate[ind], with_t=with_t[ind], act=ops.relu))
                else:
                    raise
            else:
                tmp.append(nn.Identity())
            self.ups.append(tmp)

        out_dim = default(out_dim, channels)
        # self.out_top = top_ch(rate=top1[num_resolutions])
        self.final_conv = nn.SequentialCell(
            block_klass(dim, dim, bias=bias),
            nn.Conv3d(dim, out_dim, (1, 1, 1), has_bias=bias), )
        if args.out_mean == 'START_X' and args.out_model == 'Tanh':
            self.final_conv.append(nn.Tanh())

    def construct_with_cond_scale(
            self,
            *args,
            cond_scale=2.,
            **kwargs
    ):
        logits = self.construct(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.construct(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def construct(
            self,
            x,
            time,
            cond=None,
            null_cond_prob=0.,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely
            # arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        # with autocast(enabled=self.amp):
        batch = x.shape[0]
        focus_present_mask = default(focus_present_mask,
                                     lambda: prob_mask_like((batch,), prob_focus_present))

        # time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=self.time_rel_pos_bias(x.shape[2]))
        t = self.time_mlp(time) if exists(self.time_mlp) and exists(time) else None

        # classifier free guidance

        time_rel_pos = []
        for i, (block1, block2, spatial_attn, temporal_attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            time_rel_pos.append(self.time_rel_pos_bias(x.shape[2]))
            x = temporal_attn(x, pos_bias=time_rel_pos[-1], focus_present_mask=focus_present_mask)
            if i == len(self.downs) - 1:
                h = x.copy()
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=self.time_rel_pos_bias(x.shape[2]),
                                   focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)
        x = ops.cat((x, h), axis=1)
        for i, (block1, block2, spatial_attn, temporal_attn, upsample, top) in enumerate(self.ups):
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            pos_bias = time_rel_pos.pop()
            x = temporal_attn(x, pos_bias=pos_bias, focus_present_mask=focus_present_mask)
            x1 = top(x)
            if not self.args['train']:
                if not os.path.exists(f'output/feature'):
                    os.makedirs(f'output/feature')
                if time == 0:
                    x = ops.relu(x)
                    x2 = x1[0]
                    if self.args['test'] and i == self.args['show_top_layer']:
                        x2 = top.feat_mask2(x, mask=self.args['mask'], k=self.args['k'], ns=self.args['ns'],
                                            start=self.args['start'], w=self.args['w'])
                        x2 = ops.cat([x1[0], x2], axis=0)
                    elif self.args['test']:
                        x1 = ms_load(f'output/feature/{i}_1.ckpt')
                        x2 = x * x1
                    else:
                        ms_save(x1[1], f'output/feature/{i}_1.ckpt')
                        ms_save(x, f'output/feature/{i}_.ckpt')

                    x1 = (x2,)
            if self.args['use_top'][i]:
                x = x1[0]
            else:
                x = ops.relu(x1)
            x = upsample(x)

        return self.final_conv(x)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.get_parameters(), ma_model.get_parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
