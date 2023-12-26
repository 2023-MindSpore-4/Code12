from pytorch3d.renderer.cameras import PerspectiveCameras
from nnutils import mesh_utils
from utils.contact_utils import compute_normalized_weight_querypoints
from models.layers import Transformer
import commentjson as json
import tinycudann as tcnn
import numpy as np
from mindspore import nn, ops
import mindspore as ms
def trick(x, xyz):
    within_cube = ops.all(ops.abs(xyz) <= 1, dim=-1, keepdim=True).float()  # (NP, )
    apprx_dist= .3
    x = within_cube * x + (1 - within_cube) * (apprx_dist)
    return x

def get_embedder(multires=10, **kwargs):
    if multires == -1:
        return nn.Identity(), kwargs.get('input_dims', 3)

    embed_kwargs = {
        'include_input': True,
        'input_dims': kwargs.get('input_dims', 3),
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'band_width': kwargs.get('band_width', 1),
        'periodic_fns': [ops.sin, ops.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embedder_obj, embedder_obj.out_dim


# Positional encoding (section 5.1)
class Embedder:
    """from https://github.com/yenchenlin/nerf-pytorch/blob/bdb012ee7a217bfd96be23060a51df7de402591e/run_nerf_helpers.py"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        band_width = self.kwargs['band_width']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = band_width * 2. ** ops.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = ops.linspace(band_width * 2. ** 0., band_width * 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return ops.cat([fn(inputs) for fn in self.embed_fns], -1)

    def __call__(self, inputs):
        return self.embed(inputs)

class mlp_net(nn.Cell):
    def __init__(self, in_dim=2, out_dim=1, dim=256):
        super().__init__()
        # Submodules
        self.fc_in = nn.Dense(in_dim, dim) # 319, 256
        self.fc_1 = nn.Dense(dim, dim)
        self.fc_2 = nn.Dense(dim, dim)
        self.fc_3 = nn.Dense(dim, dim)
        self.fc_out = nn.Dense(dim, out_dim)
        self.actvn = nn.LeakyReLU(0.1)
        self.actout = nn.Tanh()

    def forward(self, sdf_init, p, xPoints):
        # BP, 1
        B, P, _1 = p.shape
        p_ = p.reshape(B*P, 1)

        input = ops.cat([p_, sdf_init], axis=-1)#
        net = self.fc_in(input)

        net = self.fc_1(self.actvn(net)) + net
        net = self.fc_2(self.actvn(net)) + net
        net = self.fc_3(self.actvn(net)) + net

        sdf_2 = self.actout(self.fc_out(net)) # 
        within_cube = ops.all(ops.abs(xPoints.reshape(-1,3)) <= 1, dim=-1, keepdim=True).float()  # (NP, )
        apprx_dist= .3
        sdf_2 = within_cube * sdf_2 + (1 - within_cube) * (apprx_dist)        
        return sdf_2 

class PixCoord(nn.Cell):
    def __init__(self, cfg, z_dim, hA_dim, freq):
        super().__init__()
        J = 16
        self.cfg = cfg
        if self.cfg.CP_concatenating:
            self.net = ImplicitNetwork(z_dim+3, multires=freq, **cfg.SDF)
        elif self.cfg.distance:
            self.net = ImplicitNetwork(z_dim+J*3, multires=freq, **cfg.SDF)
        else:
            self.net = ImplicitNetwork(z_dim, multires=freq, **cfg.SDF)

            self.FC_layer = nn.SequentialCell(
                nn.Dense(304, 320),
                nn.ReLU(),
                )   #            
            self.transformer = Transformer(dim=3, kv_dim=3, depth=5, heads=6, dim_head=64,
                                        mlp_dim=6 * 64 * 2, selfatt=False)
            with open("config/config_hash.json") as f:
                config = json.load(f)                                      
            self.hash_encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config['encoding']) 
            self.input_mlp_query = nn.SequentialCell(
                nn.Dense(32, 64),
                nn.ReLU(),
                nn.Dense(64, 32))
            self.input_mlp = nn.SequentialCell(
                nn.Dense(256, 128),
                nn.ReLU(),
                nn.Dense(128, 64),
                nn.ReLU(),)   
            if self.cfg.no_pe:
                self.in_dim_mlp = 3+256+3
            else:
                self.in_dim_mlp = 3+256+32
            self.mlp_dim = self.in_dim_mlp * 2
            # self.mlp_dim = 512
            self.dp = self.cfg.dropout
            self.output_mlp = nn.SequentialCell(
                nn.Dense(self.in_dim_mlp, self.mlp_dim),
                nn.Dropout(self.dp),
                nn.Dense(self.mlp_dim, self.mlp_dim),
                nn.Dropout(self.dp),
                nn.Dense(self.mlp_dim, self.mlp_dim),
                nn.Dropout(self.dp),
                nn.Dense(self.mlp_dim, self.mlp_dim),
                nn.Dropout(self.dp),
                nn.Dense(self.mlp_dim, self.mlp_dim),
                nn.Dropout(self.dp),
                nn.Dense(self.mlp_dim, 1),
                nn.Tanh())                       
    def get_dist_joint(self, nPoints, jsTn):
        N, P, _ = nPoints.shape()
        num_j = jsTn.shape(1)
        nPoints_exp = nPoints.view(N, 1, P, 3).expand(N, num_j, P, 3).reshape(N * num_j, P, 3)
        jsPoints = mesh_utils.apply_transform(nPoints_exp, jsTn.reshape(N*num_j, 4, 4)).view(N, num_j, P, 3)
        jsPoints = jsPoints.swapaxes(1, 2).reshape(N, P, num_j * 3) # N, P, J, 3
        return jsPoints  # (N, P, J)

    # TODO: name!!
    def sample_multi_z(self, xPoints, z, cTx, cam):
        N1, P, D = xPoints.shape()
        N = z.shape(0)
        xPoints_exp = xPoints.expand(N, P, D)

        ndcPoints = self.proj_x_ndc(xPoints_exp, cTx, cam)
        zs = mesh_utils.sample_images_at_mc_locs(z, ndcPoints)  # (N, P, D)
        return zs

    def proj_x_ndc(self, xPoints, cTx, cam:PerspectiveCameras):
        cPoints = mesh_utils.apply_transform(xPoints, cTx)
        ndcPoints = mesh_utils.transform_points(cPoints, cam)
        return ndcPoints[..., :2]

    def forward(self, xPoints, z, hA, pc_target,contact_area_hand_contact, cTx=None,
                cam: PerspectiveCameras=None, jsTx=None, weight_input=None):
        N, P, _ = xPoints.shape()

        glb, local = z
        local = self.sample_multi_z(xPoints, local, cTx, cam)
        latent = glb.unsqueeze(1) + local # 256
        latent_img = latent.reshape(-1,256)

        if self.cfg.CP_concatenating: # only_feats / cat
            xPoints = xPoints.reshape(N*P,3)
            num=P-115
            if num>=0:

                addition = ops.zeros((N, num, 3)).to(xPoints.device)
                latent_hand = ops.cat([pc_target,addition], axis=1) # N, P, 3
            else:
                latent_hand = pc_target[:,:P,:]
            latent_hand = latent_hand.reshape(N*P, 3)
            latent_img_hand = ops.cat([latent_img, latent_hand], axis=-1) # N*P, 256+3
            points = ops.cat([latent_img_hand, xPoints], axis=-1)
            sdf_value = self.net(points)

            sdf_value = sdf_value.view(N, P, 1)
        elif self.cfg.random_sampling:
            transformer_output = self.transformer(xPoints, pc_target)
            points_encoding = self.hash_encoding(xPoints.reshape(N*P,3)).to(ms.float32) # [N*P, 32]
            if self.cfg.no_pe:
                input_mlp = ops.cat([transformer_output.reshape(N*P,3), latent_img, xPoints.reshape(N*P,3)], axis=-1) # N*P, 3+256+3
            else:
                input_mlp = ops.cat([transformer_output.reshape(N*P,3), latent_img, points_encoding], axis=-1) # N*P, 3+256+32
            sdf_value = self.output_mlp(input_mlp) # N*P, 3+256+32
            sdf_value = trick(sdf_value, points_encoding)
            sdf_value = sdf_value.view(N, P, 1)          
        elif self.cfg.distance: # weight_feats
            pc_space = xPoints
            weight,_ = compute_normalized_weight_querypoints(pc_space, pc_target, contact_area_hand_contact)

            glb, local = z
            local = self.sample_multi_z(xPoints, local, cTx, cam)
            # (N, P, 3) * (N, J, 12)  --> N, J, P, 3  -> N, P, J
            dstPoints = self.get_dist_joint(xPoints, jsTx)
            latent = self.cat_z_hA((glb, local, dstPoints), hA)
            latent = latent * weight
            points = self.net.cat_z_point(xPoints, latent)
            sdf_value = self.net(points)
            sdf_value = sdf_value.view(N, P, 1)            
        else:        

            transformer_output = self.transformer(xPoints, pc_target)
            points_encoding = self.hash_encoding(xPoints.reshape(N*P,3)).to(ms.float32) # [N*P, 32]
            if self.cfg.no_pe:
                input_mlp = ops.cat([transformer_output.reshape(N*P,3), latent_img, xPoints.reshape(N*P,3)], -1) # N*P, 3+256+3
            else:
                input_mlp = ops.cat([transformer_output.reshape(N*P,3), latent_img, points_encoding], -1) # N*P, 3+256+32
            sdf_value = self.output_mlp(input_mlp) # N*P, 3+256+32
            sdf_value = trick(sdf_value, points_encoding)
            sdf_value = sdf_value.view(N, P, 1)

        return sdf_value

    def cat_z_hA(self, z, hA):
        glb, local, dst_points = z
        out = ops.cat([(glb.unsqueeze(1) + local), dst_points], -1)
        return out


class ImplicitNetwork(nn.Cell):
    def __init__(
            self,
            latent_dim,
            feature_vector_size=0,
            d_in=3,
            d_out=1,
            DIMS=[ 512, 512, 512, 512, 512, 512, 512, 512 ], 
            GEOMETRIC_INIT=False,
            bias=1.0,
            SKIP_IN=(4, ),
            weight_norm=True,
            multires=10,
            th=True,
            **kwargs
    ):
        self.xyz_dim = d_in
        super().__init__()
        dims = [d_in + latent_dim] + list(DIMS) + [d_out + feature_vector_shape]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch + latent_dim

        self.num_layers = len(dims)
        self.skip_in = SKIP_IN
        self.layers = nn.CellDict()
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Dense(dims[l], out_dim)

            if GEOMETRIC_INIT:
                if l == self.num_layers - 2:
                    ops.uniform(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    ops.uniform(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # torch.nn.init.constant_(lin.weight[:, 0:latent_dim], 0.0)
                    torch.nn.init.constant_(lin.weight[:, latent_dim+3:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            # setattr(self, "lin" + str(l), lin)
            self.layers.add_module("lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        if th:
            self.th = nn.Tanh()
        else:
            self.th = nn.Identity()
        
    def forward(self, input, compute_grad=False):
        xyz = input[:, -self.xyz_dim:]
        latent = input[:, :-self.xyz_dim]

        if self.embed_fn is not None:
            xyz = self.embed_fn(xyz)
        input = ops.cat([latent, xyz], 1)
        x = input

        for l in range(0, self.num_layers - 1):
            # lin = getattr(self, "lin" + str(l))
            lin = self.layers["lin" + str(l)]

            if l in self.skip_in:
                x = ops.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        x = self.th(x)
        within_cube = ops.all(ops.abs(xyz) <= 1, dim=-1, keepdim=True).float()  # (NP, )
        # 
            # apprx_dist= (torch.norm(xyz, dim=-1, keepdim=True) - 1).clamp(min=.3)
        apprx_dist= .3
        x = within_cube * x + (1 - within_cube) * (apprx_dist)
        return x

    def gradient(self, x, sdf=None):
        """
        :param x: (sumP, D?+3)
        :return: (sumP, 1, 3)
        """
        x.requires_grad_(True)
        if sdf is None:
            y = self.forward(x)[:, :1]
        else:
            y = sdf(x)
        d_output = ops.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # only care about xyz dim
        gradients = gradients[..., -3:]

        # only care about sdf within cube
        xyz = x[..., -self.xyz_dim:]
        within_cube = ops.all(ops.abs(xyz) < 1, dim=-1, keepdim=True).float()  # (NP, )
        gradients = within_cube * gradients + (1 - within_cube) * 1 / np.sqrt(gradients.shape(-1))
        return gradients.unsqueeze(1)

    def cat_z_point(self, points, z):
        """
        :param points: (N, P, 3)
        :param z: (N, (P, ), D)
        :return: (NP, D+3)
        """
        if z.ndim == 3:
            N, P, D = z.shape()
            return ops.cat([z, points], -1).reshape(N*P, D+3)
        N, D = z.shape()
        if points.ndim == 2:
            points = points.unsqueeze(0)
        NP, P, _ = points.shape()
        assert N == NP

        z_p = ops.cat([z.unsqueeze(1).repeat(1, P, 1), points], -1)
        z_p = z_p.reshape(N * P, D + 3)
        return z_p


class PixObj(PixCoord):
    def __init__(self, cfg, z_dim, hA_dim, freq):
        super().__init__(cfg, z_dim, hA_dim, freq)
        J = 16
        self.net = ImplicitNetwork(z_dim, multires=freq, 
            **cfg.SDF)
    
    def cat_z_hA(self, z, hA):
        glb, local, _ = z 
        glb = glb.unsqueeze(1)
        return glb + local


def build_net(cfg, z_dim=None):
    if z_dim is None:
        z_dim = cfg.Z_DIM
    if cfg.DEC == 'obj':
        dec = PixObj(cfg, z_dim, cfg.THETA_DIM, cfg.FREQ)
    else:
        dec = PixCoord(cfg, z_dim, cfg.THETA_DIM, cfg.FREQ)
    return dec

