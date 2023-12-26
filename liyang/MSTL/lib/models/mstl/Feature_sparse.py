""""
This page will be released soon
"""
from mindspore import nn
import math
from lib.models.stark.position_encoding import PositionEmbeddingLearned_new
class MHCA_FS(nn.Cell):

    def __init__(self, dim=128 , num_heads=8,num_embeddings=32,embedding_dim=128,
                 drop=0., norm_layer=nn.LayerNorm,seq_len = 16,use_pos_emd_in =True, pos_enc = True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim = dim, num_heads = num_heads, dropout=drop)
        self.dropout1 = nn.Dropout(drop)
        self.pos_emd_out = PositionEmbeddingLearned_new(dim // 2, int( math.sqrt(num_embeddings) ))
        self.seq_in_w = seq_len
        self.seq_out_w = int( math.sqrt(num_embeddings) )
        self.dim = embedding_dim
        self.use_pos_emd_in = use_pos_emd_in
        if use_pos_emd_in:
            self.pos_emd_in = PositionEmbeddingLearned_new(dim // 2, seq_len)
        self.FS = nn.Embedding(num_embeddings, embedding_dim)

        self.use_pos_enc = pos_enc
        if pos_enc:
            self.pos_enc_emb = nn.Embedding(1, embedding_dim)


    def forward(self, k,v,key_padding_mask=None):
        #x should be ( 2,5*32*32,256),which is (WHT,B,C)
        THW, B, C = k.shape

        # q is the proposed 'general object template'
        q = self.FS.weight.unsqueeze(1).tile(1, B, 1)

        # attention operations
        q = q + self.dropout1(self.attn(query=q , key=k, value=v, key_padding_mask=key_padding_mask)[0])
        q = self.norm1(q)
        return q