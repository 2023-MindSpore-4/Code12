from math import pi

import mindspore
from mindspore import Tensor, Parameter, nn, ops
from mindspore.ops import einsum


# helper functions

def exists(val):
    return val is not None


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return mindspore.ops.cat(tensors, axis=dim)


# rotary embedding helper functions

def rotate_half(x):
    # x = rearrange(x, '... (d r) -> ... d r', r=2)
    shape = list(x.shape)
    shape = shape[:-1]
    shape.append(-1)
    shape.append(2)
    x = x.view(tuple(shape))
    x1, x2 = x.unbind(dim=-1)
    # x1, x2 = ops.unbind(x,dim=-1)
    x = mindspore.ops.stack((-x2, x1), axis=-1)
    # return rearrange(x, '... d r -> ... (d r)')
    shape = list(x.shape)
    shape = shape[:-2]
    shape.append(-1)
    x = x.reshape(tuple(shape))
    return x


def apply_rotary_emb(freqs, t, start_index=0, scale=1., seq_dim=-2):
    rot_dim, seq_len = freqs.shape[-1], t.shape[seq_dim]
    freqs = freqs[-seq_len:].to(t.dtype)

    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[
        -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return ops.cat((t_left, t, t_right), axis=-1)


#
# def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
#     freqs = freqs.to(t.dtype)
#     rot_dim = freqs.shape[-1]
#     end_index = start_index + rot_dim
#     assert rot_dim <= t.shape[
#         -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
#     t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
#     t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
#     return mindspore.ops.cat((t_left, t, t_right), axis=-1)


# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        shape = rotations.shape
        shape = list(shape)
        shape = shape[:-2]
        shape.append(-1)
        shape = tuple(shape)
        rotations = rotations.reshape(shape)
        # rotations = rearrange(rotations, '... r f -> ... (r f)')

    # rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    rotations = mindspore.ops.expand_dims(rotations, -1)
    rotations = mindspore.ops.repeat_elements(rotations, 2, -1)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes

class RotaryEmbedding(nn.Cell):
    def __init__(
            self,
            dim,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
            use_xpos=False,
            xpos_scale_base=512,
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (mindspore.ops.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = mindspore.ops.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = mindspore.ops.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = Parameter(freqs, requires_grad=learned_freq)

        self.use_xpos = use_xpos
        if not use_xpos:
            self.scale = None
            return

        scale = (mindspore.ops.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.insert_param_to_cell('scale', scale)

    def rotate_queries_or_keys(self, t, seq_dim=-2):
        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
        seq_len = t.shape[seq_dim]
        freqs = self.forward(lambda: mindspore.ops.arange(seq_len), cache_key=seq_len)
        return apply_rotary_emb(freqs, t)

    def rotate_queries_and_keys(self, q, k, seq_dim=-2):
        assert self.use_xpos
        device, seq_len = q.device, q.shape[seq_dim]
        seq = mindspore.ops.arange(seq_len)
        freqs = self.forward(lambda: seq, cache_key=seq_len)
        scale = self.get_scale(lambda: seq, cache_key=seq_len)
        rotated_q = apply_rotary_emb(freqs, q, scale=scale)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1)
        return rotated_q, rotated_k

    # @cache
    def get_scale(self, t, cache_key=None):
        assert self.use_xpos

        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        scale = 1.
        if self.use_xpos:
            power = t - (len(t) // 2) / self.scale_base
            power = Tensor(power).expand_dims(axis=-1)
            scale = self.scale ** power
            scale = mindspore.ops.cat((scale, scale), axis=-1)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return scale

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.float(), freqs)
        # freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = mindspore.ops.repeat_elements(freqs, 2, -1)
        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs
