import math

import mindspore
import numpy as np
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import initializer
from mindspore.ops import einsum


def exists(val):
    return val is not None


class Attention(nn.Cell):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None,
            dtype=mindspore.float32
    ):
        super().__init__()
        self.reshape = ops.Reshape()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Dense(dim, hidden_dim * 3, has_bias=False).to_float(dtype)
        self.to_out = nn.Dense(hidden_dim, dim, has_bias=False).to_float(dtype)

    def construct(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):

        qkv = ops.chunk(self.to_qkv(x), 3, axis=-1)

        def rearange_in(x1):
            h = self.heads
            b, t, n, d = x1.shape
            d = d // h

            x1 = self.reshape(x1, (b, t, n, h, d))
            x1 = self.transpose(x1, (0, 1, 3, 2, 4))
            return x1

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = rearange_in(q)
        k = rearange_in(k)
        v = rearange_in(v)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask):
            focus_present_mask = self.reshape(focus_present_mask, (focus_present_mask.shape[0], -1))
            if sim.dtype == mindspore.float16:
                finfo_type = np.float16
            else:
                finfo_type = np.float32
            max_neg_value = -np.finfo(finfo_type).max
            focus_present_mask = focus_present_mask.to(mindspore.int16)
            focus_present_mask = focus_present_mask.repeat(self.heads, axis=0)
            focus_present_mask = ops.expand_dims(focus_present_mask, axis=1)
            sim.masked_fill(focus_present_mask.to(mindspore.bool_), max_neg_value)

        # numerical stability

        sim = sim - ops.amax(sim, axis=1, keepdims=True)
        attn = self.softmax(sim)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = ops.swapaxes(out, -2, -3)
        sh = out.shape[:-2]
        out = out.view(*sh, -1)
        return self.to_out(out)


class RelativePositionBias(nn.Cell):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).int() * num_buckets
        n = ops.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                ops.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = ops.minimum(val_if_large, ops.full_like(val_if_large, num_buckets - 1))
        # val_if_large = ops.where(val_if_large < num_buckets - 1, val_if_large, num_buckets - 1)
        ret += ops.where(is_small, n, val_if_large)
        return ret

    def construct(self, n):
        q_pos = ops.arange(n, dtype=mindspore.int64)
        k_pos = ops.arange(n, dtype=mindspore.int64)
        rel_pos = k_pos.view(1, -1) - q_pos.view(-1, 1)
        # rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return values.transpose(2, 0, 1)
        # return rearrange(values, 'i j h -> h i j')


# class LayerNorm(nn.cell):
#     def __init__(self, dim, eps=1e-3):
#         super().__init__()
#         self.eps = eps
#         self.gamma = nn.Parameter(ops.ones(1, dim, 1, 1, 1))
#
#     def construct(self, x):
#         var = ops.var(x, axis=1, keepdims=True)
#         mean = ops.mean(x, axis=1, keep_dims=True)
#         return (x - mean) / (var + self.eps).sqrt() * self.gamma
def rsqrt(x):
    res = ops.sqrt(x)
    return ops.inv(res)


class LayerNorm(nn.Cell):
    def __init__(self, dim, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(initializer('ones', (1, dim, 1, 1, 1)))

    def construct(self, x):
        var = x.var(1, keepdims=True)
        mean = x.mean(1, keep_dims=True)
        return (x - mean) * rsqrt((var + self.eps)) * self.gamma


class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

        self.norm = LayerNorm(dim)

    def construct(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Block(nn.Cell):
    def __init__(self, dim, dim_out, groups=8, bias=True):
        super().__init__()
        self.bias = bias
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), pad_mode='pad', padding=(0, 0, 1, 1, 1, 1), has_bias=bias)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def construct(self, x, scale_shift=None):
        x = self.proj(x)
        if self.bias:
            shape = x.shape
            x = x.view(*shape[:-2], -1)
            x = self.norm(x)
            x = x.view(*shape)

        if exists(scale_shift):
            scale, shift = scale_shift
            x *= (scale + 1)
            if self.bias:
                x += shift

        return self.act(x)


class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, *args, **kwargs):
        a = self.fn(x, *args, **kwargs)
        return a + x


class SinusoidalPosEmb(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = ops.exp(ops.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = ops.cat((emb.sin(), emb.cos()), axis=-1)
        return emb


class ResnetBlock(nn.Cell):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, bias=True):
        super(ResnetBlock, self).__init__()

        d = dim_out * 2
        self.mlp = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(time_emb_dim, d, has_bias=bias)
        ) if exists(time_emb_dim) else None
        self.bias = bias
        self.block1 = Block(dim, dim_out, groups=groups, bias=bias)
        self.block2 = Block(dim_out, dim_out, groups=groups, bias=bias)
        self.res_conv = nn.Conv3d(dim, dim_out, (1, 1, 1), pad_mode='pad',
                                  has_bias=bias) if dim != dim_out else nn.Identity()

    def construct(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            b, c = time_emb.shape
            time_emb = time_emb.view(b, c, 1, 1, 1)
            # time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, axis=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


def rearrange_s(head, inputs):
    b, hc, x, y = inputs.shape
    c = hc // head
    return inputs.view(b, head, c, x * y)


class SpatialLinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32, bias=True):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, (1, 1), pad_mode='pad', has_bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, (1, 1), pad_mode='pad', has_bias=bias)
        self.map = ops.Map()
        self.partial = ops.Partial()

    def construct(self, x):
        b, c, f, h, w = x.shape
        x = x.swapaxes(1, 2).view(b * f, c, h, w)
        # x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, axis=1)
        # q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)
        q, k, v = self.map(self.partial(rearrange_s, self.heads), qkv)

        q = ops.softmax(q, axis=-2)
        k = ops.softmax(k, axis=-1)

        q = q * self.scale
        context = ops.einsum('b h d n, b h e n -> b h d e', k, v)

        out = ops.einsum('b h d e, b h d n -> b h e n', context, q)
        shape = out.shape
        # out = out.view(shape[0], self.heads, shape[2], h, w)
        out = out.view(shape[0], self.heads * shape[2], h, w)
        # out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return out.view(b, f, c, h, w).swapaxes(1, 2)
        # return rearrange(out, '(b f) c h w -> b c f h w', b=b)


def Upsample(dim, bias=True):
    return nn.Conv3dTranspose(dim, dim, (4, 4, 4), (2, 2, 2), padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=bias)


def Downsample(dim, bias=True):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=bias)


def prob_mask_like(shape, prob):
    if prob == 1:
        return ops.ones(shape, dtype=mindspore.bool_)
    elif prob == 0:
        return ops.zeros(shape, dtype=mindspore.bool_)
    else:
        return ops.zeros(shape).float().uniform_(0, 1) < prob


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather_elements(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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


def mask_4d(x, mask=None, k=None):
    tmp = ops.ones_like(x)
    if not mask[0] is None:
        tmp2 = ops.zeros_like(x)
        tmp2[..., mask[0]:mask[0] + k[0], :, :, :] = 1
        tmp = tmp2 * tmp
    if not mask[1] is None:
        tmp2 = ops.zeros_like(x)
        tmp2[..., mask[1]:mask[1] + k[1], :, :] = 1
        tmp = tmp2 * tmp
    if not mask[2] is None:
        tmp2 = ops.zeros_like(x)
        tmp2[..., mask[2]:mask[2] + k[2], :] = 1
        tmp = tmp2 * tmp
    if not mask[3] is None:
        tmp2 = ops.zeros_like(x)
        tmp2[..., mask[3]:mask[3] + k[3]] = 1
        tmp = tmp2 * tmp
    return tmp


class top_ch(nn.Cell):
    def __init__(self, min_k=1, dim=1, rate=0.1, largest=True, sort_ed=True, act=ops.relu, shape=None, eps=1e-6):
        super().__init__()
        self.act = act
        self.min_k = min_k
        self.dim = dim
        self.rate = rate
        self.largest = largest
        self.sort_ed = sort_ed
        self.shape = shape
        self.eps = eps

    def construct(self, x, return_indices=True):
        x = self.act(x)
        _, indices = x.topk(
            k=max(int(x.size(self.dim) * self.rate), self.min_k),
            dim=self.dim,
            largest=self.largest,
            sorted=self.sort_ed,
        )
        tmp = ops.zeros_like(x)
        tmp = tmp.scatter(self.dim, indices, 1.)
        x = tmp * x
        return x, tmp if return_indices else x

    def feat_mask(self, x, mask=None, k=None, ns=24, start=0, w=1):
        x = self.act(x)
        x *= mask_4d(x, mask, k)
        _, indices = x.topk(
            k=x.size(self.dim),
            dim=self.dim,
            largest=self.largest,
            sorted=self.sort_ed
        )
        if ns > x.size(self.dim):
            ns = x.size(self.dim)
            start = 0
        x1 = ops.zeros(0).to(x.device)

        for n in range(ns):
            tmp = ops.zeros_like(x)
            tmp = tmp.scatter(self.dim, get_indi(indices=indices, n=n + ns * start, w=w, dim=self.dim), 1.)
            x1 = ops.cat((x1, tmp * x), axis=0)
        return x1


class top_txy(nn.Cell):
    def __init__(self, min_k=1, rate=0.1, with_t=False, largest=True, sort_ed=True, act=ops.relu, shape=None, eps=1e-6):
        super().__init__()
        self.act = act
        self.min_k = min_k
        self.rate = rate
        self.largest = largest
        self.sort_ed = sort_ed
        self.shape = shape
        self.with_t = with_t
        self.dim = -1
        self.eps = eps

    def construct(self, x, return_indices=True):
        size = x.shape
        x = self.act(x)
        if self.with_t:
            x = x.reshape(size[0], size[1], -1)
        else:
            x = x.reshape(size[0], size[1], size[2], -1)
        _, indices = x.topk(
            k=max(int(x.shape[-1] * self.rate), self.min_k),
            dim=self.dim,
            largest=self.largest,
            sorted=self.sort_ed
        )
        tmp = ops.zeros_like(x)
        tmp = tmp.scatter(self.dim, indices, ops.ones_like(indices, dtype=tmp.dtype)).view(size)
        x = x.view(size)
        x = tmp * x
        #
        return x, ops.where(x > 0, ops.ones_like(x), ops.zeros_like(x)) if return_indices else x

    def feat_mask(self, x, mask=None, k=None, ns=24, start=0, w=1):
        size = x.shape
        x = self.construct(x)[0]
        x *= mask_4d(x, mask, k)
        # if self.with_t:
        #     x = x.reshape(size[0], size[1], -1)
        # else:
        #     x = x.reshape(size[0], size[1], size[2], -1)
        dim = 1
        _, indices = x.topk(
            k=x.size(dim),
            dim=dim,
            largest=self.largest,
            sorted=self.sort_ed
        )
        if ns > x.size(1):
            ns = x.size(1)
            start = 0
        x1 = ops.zeros(0).to(x.device)
        for n in range(ns):
            tmp = ops.zeros_like(x)
            tmp = tmp.scatter(dim, get_indi(indices=indices, n=n + ns * start, w=w, dim=dim),
                              ops.ones_like(indices, dtype=tmp.dtype))
            x1 = ops.cat((x1, (tmp * x).view(size)), axis=0)
        return x1

    def feat_mask2(self, x, mask=None, k=None, ns=24, start=0, w=1):
        size = x.shape
        x = self.construct(x)[0]
        x *= mask_4d(x, mask, k)

        X = x.sum(axis=2, keepdims=True)
        X = X.sum(axis=3, keepdims=True)
        X = X.sum(axis=4, keepdims=True)

        dim = 1
        _, indices1 = X.topk(
            k=X.shape[dim],
            dim=dim,
            largest=self.largest,
            sorted=self.sort_ed
        )
        x1 = None
        for n in range(ns):
            tmp = ops.zeros_like(X, dtype=X.dtype)
            now = get_indi(indices=indices1, n=n + ns * start, w=w, dim=dim)
            tmp = tmp.scatter(dim, now, ops.ones_like(now, dtype=tmp.dtype))
            tmp = tmp * x
            tmp = tmp.view(size[0], size[1], size[2], -1)
            _, indices = tmp.topk(
                k=1,
                dim=3,
                largest=self.largest,
                sorted=self.sort_ed,
            )
            tmp2 = ops.zeros_like(tmp, dtype=tmp.dtype)
            tmp2 = tmp2.scatter(3, indices, ops.ones_like(indices, dtype=tmp.dtype))
            if x1 is None:
                x1 = (tmp2 * tmp).view(*size)
            else:
                x1 = ops.cat((x1, (tmp2 * tmp).view(*size)), axis=0)
        return x1


def top_(x, k, dim=-1, largest=True, sort_ed=True, act=ops.relu, ):
    size = x.size()
    x = act(x)
    x = x.view(size[0], size[1], -1)
    _, indices = x.topk(k, dim=dim, largest=largest, sorted=sort_ed)
    tmp = ops.zeros_like(x)
    x = tmp.scatter(dim, indices, 1.) * x
    return x.view(size)
