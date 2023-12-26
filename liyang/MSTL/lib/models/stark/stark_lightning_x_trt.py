"""
2021.06.24 STARK Lightning-X Model (Spatial-only).
2021.06.27 for converting pytorch model to trt model
"""
import torch
from torch import nn
import torch.nn.functional as F
from .backbone_X import build_backbone_x
from .position_encoding import build_position_encoding_new
from .lite_encoder import build_lite_encoder,build_lite_encoders  # encoder only
from .head import build_box_head,MLP
from lib.utils.box_ops import box_xyxy_to_cxcywh
import time
from lib.models.stark.TemperalAdaptive import Tada
from lib.models.TImesFormer.Feature_sparse import MHCA_FS
from lib.utils.merge import get_qkv,get_space_time_qkv,get_FS_qkv,merge_template_search
from einops import rearrange
from plot_features.visual712 import PLT_MAP,plot_map
import math
from .transformer import build_transformer
from lib.models.loss.filter import apply_filter
from lib.utils.load_pretrained import load_pretrain_model
from lib.models.stark.TemperalAdaptive import Tada

class STARKLightningXtrt(nn.Cell):
    """Modified from stark_s_plus_sp
    The goal is to achieve ultra-high speed (1000FPS)
    2021.06.24 We change the input datatype to standard Tensor, rather than NestedTensor
    2021.06.27 Definition of transformer is changed"""
    def __init__(self, backbone, transformer, box_head, num_template,num_search,num_sparse_temp ,num_sparse_scr, Temp_FS,pos_emb_temp_dense,
        pos_emb_scr_dense,Scr_FS ,clss_prj ,bbox_prj ,head_type="CORNER_LITE",distill=False,Tada  =None):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_search = num_search
        self.num_template = num_template
        self.num_sparse_temp = num_sparse_temp
        self.num_sparse_scr  = num_sparse_scr
        self.num_dense_temp = num_template - num_sparse_temp
        self.num_dense_scr  = num_search - num_sparse_scr
        self.pos_emb_temp_dense = pos_emb_temp_dense
        self.pos_emb_scr_dense = pos_emb_scr_dense
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(1, hidden_dim)
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.head_type = head_type
        self.distill = distill
        self.Temp_FS = Temp_FS
        self.num_stage = len(  Scr_FS)
        self.scr_stage1_num = 1
        self.scr_stage2_num = 2

        self.clss_score_emb =nn.Embedding(1, hidden_dim)
        #self.bbox_score_rate = nn.Parameter(ops.randn(1))
        #self.pos_enc_weight = nn.Parameter(ops.randn(1))

        for i in range(self.num_stage):
            setattr(self, 'Scr%d_FS'%i, Scr_FS[i] )

        self.scr_stage1_num = 1 # num for 4*4
        self.scr_stage2_num = 2 # num for 2*2

        """for backbobe"""
        self.Tada = Tada
        self.backbone_std = None
        self.backbone_score = None


        self.dec_opt = None

        #self.draw_map = PLT_MAP(2,2)
        if "CORNER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        self.clss_prj = clss_prj
        self.bbox_prj = bbox_prj


    def forward(self, img=None, mask=None, q=None, k=None, v=None, key_padding_mask=None,feat_dict_list=None,
                mode="backbone", zx="template", softmax=True,plot =None,pos_enc = None,pos_enc_emb = None):
        if mode == "backbone":
            return self.forward_backbone(img, zx, mask,plot = plot,pos_enc = pos_enc )
        elif mode[:-1] == "transformer":
            return self.forward_transformer(feat_dict_list=feat_dict_list,plot  = plot,id = int(  mode[-1] ))
        elif mode =='Temp_FS':
            return self.forward_FS(feat_dict_list = feat_dict_list,mode= mode,pos_enc_emb =pos_enc_emb  )
        elif 'Scr' in mode:
            return self.forward_FS(feat_dict_list = feat_dict_list,mode= mode , pos_enc_emb =pos_enc_emb )
        else:
            raise ValueError

    def forward_FS(self,feat_dict_list,mode,pos_enc_emb = None):
        """1.perform forward
        2. resize the mask
        3.get the sparse feature vector"""
        if mode =='Temp_FS':
            FS = self.Temp_FS
            H,W,T,S_H,S_W = FS.seq_in_w,FS.seq_in_w,1,FS.seq_out_w,FS.seq_out_w
            C = FS.dim

        elif 'Scr' in mode:
            FS =  getattr(self, mode)
            H,W,T,S_H,S_W = FS.seq_in_w,FS.seq_in_w,1,FS.seq_out_w,FS.seq_out_w
            C = FS.dim
        if FS.use_pos_emd_in:
            bs = feat_dict_list[-1]['feat'].shape[1]
            pos_in = FS.pos_emd_in(bs)
            pos_in = rearrange(pos_in, 'B C H W -> (H W) B C', B=bs, C=C, H=H, W=W)
            feat_dict_list[-1]['pos'] = pos_in
        if FS.use_pos_enc and pos_enc_emb is not None:
            #FS.pos_enc_emb.weight.reshape(1,1,-1 )
            feat_dict_list[-1]['feat'] = feat_dict_list[-1]['feat']+pos_enc_emb.unsqueeze(-1)* ( FS.pos_enc_emb.weight.reshape(1,1,-1 ) )


        k, v, mask, pos = get_FS_qkv(feat_dict_list)  # ，ith pos encoding
        bs,_ = mask.shape

        pos_out = ops.cat([FS.pos_emd_out(bs)for i in range(T)], dim=0)
        pos_out = rearrange(pos_out, "(B T) C H W -> (T H W) B C", B=bs, H=S_H, W=S_W, T=T, C=128)
        output_back= FS( k=k,v=v,key_padding_mask= mask)
        mask = rearrange(mask,"B (T H W) -> (B T) H W",B=bs,H=H,W=W,T=T)
        mask = F.interpolate(mask[None].float(), size=(S_H,S_H)).to(torch.bool)[0]
        mask = rearrange(mask, "(B T) H W -> B (T H W)", B=bs, H=S_H,W=S_W,T=T)

        return output_back,mask,pos_out

    def forward_backbone(self, img: ms.Tensor, zx: str, mask: ms.Tensor,pos_enc  =None,plot = None):
        """The input type is standard tensor
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(img, ms.Tensor)
        """run the backbone"""
        output_back = self.backbone(img)  # features & masks, position embedding for the search
        # plot_map(tensor= output_back, _type='(T B) C H W', T=1, C=192, H=output_back.shape[-1], W=output_back.shape[-1], B=output_back.shape[0])

        """get the positional encoding"""
        bs = img.shape(0)  # batch size
        if zx == "search":
            pos = self.pos_emb_scr_dense(bs)
            """run the tada"""
            self.backbone_std = output_back.clone().detach()  # store the backbone feature for next updating
            self.backbone_score = pos_enc

            output_back = self.Tada(output_back)

            """DO not use the current feature to generate features,backbone score will be updated in the boxhead"""
            if self.backbone_score is not None and self.backbone_std  is not None:
                self.Tada.update_temporal_context(input_feature=self.backbone_std, pos_enc=self.backbone_score)
            pos_enc = None

        elif zx == 'template_update':
            pos = self.pos_emb_temp_dense(bs)
            output_back = self.Tada(output_back)
            pos_enc = None
        elif "template" in zx:
            pos = self.pos_emb_temp_dense(bs)
            self.Tada.init_temporal_context(output_back,pos_enc )
            output_back = self.Tada(output_back)
            pos_enc = None
            # pos_enc = pos_enc.unsqueeze(-1) * self.query_embed.weight.reshape(1, 1, 1,-1)
            # pos_enc = pos_enc.permute(0,-1,1,2 )
        else:
            raise ValueError("zx should be 'template_0' or 'search'.")
        """get the downsampled attention mask"""
        mask_down = F.interpolate(mask[None].float(), size=output_back.shape[-2:]).to(torch.bool)[0]
        """adjust the shape"""
        return self.adjust(output_back, pos, mask_down,pos_enc)

    def forward_transformer(self, feat_dict_list,id = 0, softmax=True,plot =None,clss_head =False):
        # run the transformer encoder
        seq_dict = merge_template_search( feat_dict_list )
        enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], None,
                                                     seq_dict["pos"], mode = 'encoder', return_encoder_output=True)

        score = None

        # run the corner head
        if self.distill:
            outputs_coord, prob_tl, prob_br = self.forward_box_head(enc_mem, hs =None, softmax=softmax)
            return {"pred_boxes": outputs_coord, "prob_tl": prob_tl, "prob_br": prob_br}, None, None
        else:
            # s = time.time()
            #prj the bbox weight

            out, outputs_coord = self.forward_box_head(enc_mem,hs = None,plot =plot,score = score)
            # e = time.time()
            out['score'] = score

            # print("head time: %.1f ms" % ((e-s)*1000))
            return out, outputs_coord, enc_mem[-self.feat_len_s:]

    def forward_box_head(self, memory, softmax=True,plot =None,hs = None,score =None):
        """ memory: encoder embeddings (HW1+HW2, B, C) / (HW2, B, C)"""
        if self.head_type == "CORNER":
                # adjust shape
            if hs is not None:
                score = rearrange(score, 'C B H W -> B (H W) C ', B=score.shape[1], C=1, H=16, W=16)
                #plot_map(_type='T B (H W) C', T=1, B=2, C=1, H=16, W=16, tensor=score.unsqueeze(0))
                score = score* (self.clss_score_emb.weight.reshape(1,1,-1) )
                #plot_map(_type='T B (H W) C', T=1, B=2, C=128, H=16, W=16, tensor=score.unsqueeze(0))
                enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
                dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
                att = ops.matmul(enc_opt, dec_opt) # (B, HW, N)
                if plot is not None:
                    plot.plot(window_id=2, tensor=ops.matmul(enc_opt, dec_opt) .unsqueeze(0),
                              _type='T B (H W) C', T=1, B=1, C=1, H=16, W=16)
                    plot.plot(window_id=3, tensor=att.unsqueeze(0),
                              _type='T B (H W) C', T=1, B=1, C=1, H=16, W=16)
                    plot.plot(window_id=1, tensor= score.unsqueeze(0),
                              _type='T B (H W) C', T=1, B=1, C=1, H=16, W=16)
                opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
                """for score"""
                opt = opt + score.permute(0,2,1).unsqueeze(1)
                """end for  score"""
                bs, Nq, C, HW = opt.shape()
                opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            else:
                opt_feat = memory[-self.feat_len_s:].permute(1, 2, 0).contiguous()
                opt_feat  = opt_feat .view(*opt_feat.shape[:2], self.feat_sz_s, self.feat_sz_s).contiguous()

            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            #outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord}
            return out, outputs_coord
        elif "CORNER" in self.head_type:
            # encoder output for the search region (H_x*W_x, B, C)
            fx = memory[-self.feat_len_s:].permute(1, 2, 0).contiguous()  # (B, C, H_x*W_x)
            if plot is not None:
                plot.plot(window_id=3,tensor=fx.unsqueeze(0),_type='T B C (H W)',T=1,B=1,C=128,H=16,W=16 )
            fx_t = fx.view(*fx.shape[:2], self.feat_sz_s, self.feat_sz_s).contiguous()  # fx tensor 4D (B, C, H_x, W_x)
            # run the corner head
            if self.distill:
                coord_xyxy, prob_vec_tl, prob_vec_br = self.box_head(fx_t, return_dist=True, softmax=softmax)
                outputs_coord = box_xyxy_to_cxcywh(coord_xyxy)
                return outputs_coord, prob_vec_tl, prob_vec_br
            else:
                outputs_coord = box_xyxy_to_cxcywh(self.box_head(fx_t))
                out = {'pred_boxes': outputs_coord}
                return out, outputs_coord
        else:
            raise ValueError



    def adjust(self, src_feat: ms.Tensor, pos_embed: ms.Tensor, mask: ms.Tensor,pos_enc = None):
        """
        """
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        if pos_enc is not None:
            feat = feat + pos_enc
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}


def build_stark_lightning_x_trt(cfg, phase: str):
    """phase: 'train' or 'test'
    during the training phase, we need to
        (1) load backbone pretrained weights
        (2) freeze some layers' parameters"""
    backbone = build_backbone_x(cfg, phase=phase)
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)

    fsz_x, fsz_z = cfg.DATA.SEARCH.FEAT_SIZE, cfg.DATA.TEMPLATE.FEAT_SIZE
    fsz_z_s,fsz_x_s = cfg.MODEL.FS_TEMP.NUM,cfg.MODEL.FS_SCR.NUM



    pos_emb_temp_dense = build_position_encoding_new(cfg, fsz_z)
    pos_emb_scr_dense = build_position_encoding_new(cfg, fsz_x)

    Temp_FS = MHCA_FS(dim=128,num_embeddings= fsz_z_s,embedding_dim=128,seq_len= 8,use_pos_emd_in = False)
    Scr1_FS = MHCA_FS(dim=128, num_embeddings=16, embedding_dim=128, seq_len=16,use_pos_emd_in = True)
    Scr2_FS = MHCA_FS(dim=128, num_embeddings=4, embedding_dim=128,seq_len= 16) #change to 16 in HCTC81

    Temporal_ada = Tada(192,adaptive_feature=False)

    bbox_prj = nn.Dense( 128, 128 )
    clss_prj = nn.Dense(128, 128)


    model = STARKLightningXtrt(
        backbone,
        transformer,
        box_head,
        pos_emb_temp_dense=pos_emb_temp_dense,
        pos_emb_scr_dense = pos_emb_scr_dense,
        num_template=cfg.DATA.TEMPLATE.NUM,
        num_search = cfg.DATA.SEARCH.NUM,
        num_sparse_temp = cfg.DATA.TEMPLATE.SPARSE ,
        num_sparse_scr = cfg.DATA.SEARCH.SPARSE  ,
        head_type=cfg.MODEL.HEAD_TYPE,
        distill=cfg.TRAIN.DISTILL,
        Temp_FS  =Temp_FS ,
        Scr_FS = [Scr1_FS , Scr2_FS],
        bbox_prj = bbox_prj,
        clss_prj =clss_prj,
        Tada = Temporal_ada
    )
    #load_pretrain_model(model = model ,path = '/home/suixin/MyPaper/Code/HTCT/HTCT81/None/pretrained/STARKLightningXtrt_ep0200.pth.tar' )
    # eval_model(model)

    return model


def eval_model(model):
    # for para in model.parameters():
    #     print(para.nelement())
    ori = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ALL_ori = sum(p.numel() for p in model.parameters() )
    print('All:',int(ALL_ori/1024),'kB' )
    dict = {}
    for name in model.state_dict():
        #print(name, model.state_dict()[name].numel())
        _type = name.split('.')[0]
        if _type in dict:
            dict[_type] += model.state_dict()[name].numel()
        else:
            dict[_type] = model.state_dict()[name].numel()
    for i in dict:
        print(i,':', int( dict[i]/1024 ) ,'KB' )