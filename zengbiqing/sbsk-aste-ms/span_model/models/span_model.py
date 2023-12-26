import logging
from typing import Dict, List, Optional, Union
import copy

import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import msadapter.pytorch.nn.functional as F

from ms_allennlp_fix.type_define import (
    Vocabulary, Model, Params,
    InitializerApplicator, RegularizerApplicator
)
from ms_allennlp_fix.modules import (
    FeedForward, TimeDistributed, TextFieldEmbedder,
    EndpointSpanExtractor, SelfAttentiveSpanExtractor,SpanExtractor,
)
from ms_allennlp_fix.token_indexer_type import (
    PreTrainTransformerTokenIndexer
)
import ms_allennlp_fix.util as util

from span_model.models.ner import NERTagger
from span_model.models.relation_proper import ProperRelationExtractor
from span_model.models.shared import BiAffineSpanExtractor
from span_model.models.SelfAttention import SelfAttention
import nltk
import numpy as np
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from span_model.models.agcn import TypeGraphConvolution

# New
from msadapter.pytorch import Tensor

def global_max_pool1d(x: torch.Tensor) -> torch.Tensor:
    bs, seq_len, features = x.shape
    x = x.transpose(-1, -2)
    x = torch.nn.functional.adaptive_max_pool1d(x, output_size=1, return_indices=False)
    x = x.transpose(-1, -2)
    x = x.squeeze(dim=1)
    assert tuple(x.shape) == (bs, features)
    return x

def global_avg_pool1d(x: torch.Tensor) -> torch.Tensor:
    bs, seq_len, features = x.shape
    x = x.transpose(-1, -2)
    x = torch.nn.functional.adaptive_avg_pool1d(x, output_size=1)
    x = x.transpose(-1, -2)
    x = x.squeeze(dim=1)
    assert tuple(x.shape) == (bs, features)
    return x

class TagEmbedder(torch.nn.Module):
    '''POS Tagging'''
    def __init__(self,num_width_embeddings,
                dim=64, vocab_size=11,is_attention = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedder = torch.nn.Embedding(self.vocab_size, self.dim, padding_idx=0)
        self.tag_dict = {
            'RB':1,'RBR':1,'RBS':1,'VB':2,'VBD':2,'VBG':2,'VBN':2,'VBP':2,'VBZ':2,'JJ':3,'JJR':3,'JJS':3,
            'NN':4,'NNS':4,'NNP':4,'NNPS':4,'IN':5,'DT':6,'CC':7, 'CD':8,'RP':9
        }
        self.is_attention = is_attention
        if self.is_attention:
            self.attention = SelfAttention(dim,2,0.2)

    def forward(self, tag_tensor) -> torch.Tensor:
        tag_embedder = self.embedder(tag_tensor)
        org_shape = tag_embedder.shape  # [bs, span_num, max_span_width, hidden_dim]

        tag_embedder = tag_embedder.view(-1, org_shape[-2], org_shape[-1])

        if self.is_attention:
            tag_mask = torch.ones((tag_embedder.shape[0], tag_embedder.shape[1]))
            tag_embedder = self.attention(tag_embedder, tag_mask)
            tag_embeddings = torch.sum(tag_embedder, dim=1) # [bs * span_num, hidden_dim]
        else:
            tag_embeddings = global_max_pool1d(tag_embedder)

        tag_embeddings = tag_embeddings.reshape(
            org_shape[0], org_shape[1], org_shape[-1]
        )   # [bs, span_num, hidden_dim]
        return tag_embeddings


class TreeEmbedder(torch.nn.Module):
    '''Build Constituent Syntax Tree'''
    def __init__(self,
                dim=64, is_attention = True):
        super().__init__()
        # in other place util.py
        self.tag_list = ["RBS", "SYM", "RP", "-LRB-", "VBP", "NNS", "RRC", "PP", "NP-TMP", "EX", "NX", "SQ", "POS",
                         "DT", "NNP", "WDT", "FW", "LST", "#", ",", "QP", "ADVP", "WP", "VBD", "NNPS", "LS", "SBAR",
                         "PDT", "RBR", "UH", ".", "VBN", "TO", "X", "WHPP", "SINV", "VBZ", "SBARQ", "RB", "INTJ", "UCP",
                         "JJR", "IN", "CC", ":", "JJ", "CD", "''", "$", "PRP$", "NN", "WHADJP", "VB", "MD", "NP", "PRT",
                         "CONJP", "WRB", "VBG", "FRAG", "-RRB-", "ADJP", "WHNP", "WHADVP", "``", "PRN", "JJS", "PRP",
                         "VP"]
        self.tag_dict = dict(zip(self.tag_list, [i for i in range(1, len(self.tag_list)+1)]))

        self.vocab_size = len(self.tag_dict)+1
        self.dim = dim
        self.embedder = torch.nn.Embedding(self.vocab_size, self.dim, padding_idx=0)
        self.is_attention = is_attention
        if self.is_attention:
            self.attention1 = SelfAttention(dim, 2, 0.2)
            self.attention2 = SelfAttention(dim, 2, 0.2)

    def forward(self, tree_info, spans:Tensor) -> Tensor:
        # tree_info encode outside of model

        from fix.debug import fshape, fprint

        tag_embedding = self.embedder(tree_info)    # [bs, span_num, max_span_width, max_path_dep, embed_dim]
        batch_size = tree_info.shape[0]
        tag_embedding = tag_embedding.reshape(-1, tag_embedding.shape[3], tag_embedding.shape[4])   
        # [bs * span_num * max_span_width, max_path_dep, embed_dim]

        if self.is_attention:
            tree_mask1 = torch.ones(tag_embedding.shape[0], tag_embedding.shape[1])
            tag_embedding = torch.sum(
                self.attention1(tag_embedding, tree_mask1), dim=1
            ).unsqueeze(0)      # [1, bs * span_num * max_span_width, embed_dim]
            tag_embedding = tag_embedding.reshape(
                spans.shape[1] * batch_size, -1, tag_embedding.shape[2]
            )                   # [span_num * bs, max_span_width, embed_dim]
            tree_mask2 = torch.ones(tag_embedding.shape[0], tag_embedding.shape[1])
            tag_embedding = torch.sum(
                self.attention2(tag_embedding, tree_mask2), dim=1
            ).unsqueeze(0)      # [1, bs * span_num, embed_dim]
        else:
            tag_embedding = global_max_pool1d(
                global_avg_pool1d(
                    tag_embedding
                ).reshape(spans.shape[1], -1, tag_embedding.shape[2])
            ).unsqueeze(0)
        
        tag_embedding = tag_embedding.reshape(batch_size, -1, tag_embedding.shape[-1])    
        # [bs, span_num, embed_dim]

        return tag_embedding


class SpanWidthEmbedder(torch.nn.Module):
    '''跨度宽度嵌入'''
    def __init__(self, dim=20, num_width_embeddings=8):
        super().__init__()
        self.vocab_size = num_width_embeddings
        self.dim = dim
        self._span_width_embedding = torch.nn.Embedding(
            num_embeddings=num_width_embeddings, embedding_dim=dim,
        )

    def forward(self, spans, span_embeddings) -> torch.Tensor:
        span_widths = spans[:,:,1] - spans[:,:,0]
        span_width_embeddings = self._span_width_embedding(span_widths)
        new_span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)
        return new_span_embeddings


class MaxPoolSpanExtractor(SpanExtractor):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    @staticmethod
    def extract_pooled(x, mask) -> Tensor:
        return util.masked_max(x, mask, dim=-2)

     
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.Tensor = None,
    ) -> Tensor:
        span_embeddings, span_mask = util.batched_span_select(sequence_tensor, span_indices)
        bs, num_spans, span_width, size = span_embeddings.shape
        span_mask = span_mask.view(bs, num_spans, span_width, 1)
        x = self.extract_pooled(span_embeddings, span_mask)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            x *= span_indices_mask.view(bs, num_spans, 1)

        assert tuple(x.shape) == (bs, num_spans, size)
        return x


class MeanPoolSpanExtractor(MaxPoolSpanExtractor):
    @staticmethod
    def extract_pooled(x, mask) -> Tensor:
        return util.masked_mean(x, mask, dim=-2)


class TextEmbedderWithBiLSTM(TextFieldEmbedder):
    def __init__(self, embedder: TextFieldEmbedder, hidden_size: int):
        super().__init__()
        self.embedder = embedder
        self.lstm = torch.nn.LSTM(
            input_size=self.embedder.get_output_dim(),
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=1,  # Increasing num_layers can help but we want fair comparison
        )
        self.dropout = torch.nn.Dropout(p=0.5)
        self.output_size = hidden_size * 2

    def get_output_dim(self) -> int:
        return self.output_size

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = self.embedder(*args, **kwargs)
        x = x.squeeze(dim=0)  # For some reason x.shape is (1, 1, seq_len, size)
        x = self.dropout(x)  # Seems to work best if dropout both before and after lstm
        x, state = self.lstm(x)
        x = self.dropout(x)
        x = x.unsqueeze(dim=0)
        return x


class SpanModel(Model):
    def __init__(
        self,
        pretrain_transformer_indexers: Dict[str, PreTrainTransformerTokenIndexer],
        embedder: TextFieldEmbedder,
        modules,  # TODO: Add type.
        feature_size: int,
        max_span_width: int,
        target_task: str,
        feedforward_params: Dict[str, Union[int, float]],
        loss_weights: Dict[str, float],
        initializer: InitializerApplicator = InitializerApplicator(),
        module_initializer: InitializerApplicator = InitializerApplicator(),
        # TODO 在本模型中应用正则化
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
        # New
        use_ner_embeds: bool = None,
        span_extractor_type: str = None,
        use_double_mix_embedder: bool = None,
        relation_head_type: str = "base",
        use_span_width_embeds: bool = None,
        use_bilstm_after_embedder: bool = False,
        use_tag_embeds: bool = True,
        use_tree_embeds: bool = True,
        use_span_width: bool = False,
        use_dep: bool=True,
        dep_dim:int=64,
        len_dep_type:int=42,
        use_filter:bool=True,
        data_set_name: str = None,
    ) -> None:
        super(SpanModel, self).__init__()

        # New
        info = dict(
            use_ner_embeds=use_ner_embeds,
            span_extractor_type=span_extractor_type,
            use_double_mix_embedder=use_double_mix_embedder,
            relation_head_type=relation_head_type,
            use_span_width_embeds=use_span_width_embeds,
        )
        for k, v in info.items():
            # print(dict(locals=(k, v)))
            assert v is not None, k
        self.use_double_mix_embedder = use_double_mix_embedder
        self.relation_head_type = relation_head_type
        self.use_tag_embeds = use_tag_embeds
        self.use_tree_embeds = use_tree_embeds
        self.use_span_width = use_span_width
        self.use_filter = use_filter
        if use_bilstm_after_embedder:
            embedder = TextEmbedderWithBiLSTM(embedder, hidden_size=300)

        ####################
        modules = Params(modules)

        token_embed_dim = embedder.get_output_dim()
        # 模型测试后最优方法为使用 endpoint
        assert span_extractor_type in {"endpoint", "attn", "max_pool", "mean_pool", "bi_affine"}
        # Create span extractor.
        if use_span_width_embeds:
            self._endpoint_span_extractor = EndpointSpanExtractor(
                token_embed_dim ,
                combination="x,y",
                num_width_embeddings=max_span_width,
                span_width_embedding_dim=feature_size,
                bucket_widths=False,
            )
        # New
        else:
            self._endpoint_span_extractor = EndpointSpanExtractor(
                token_embed_dim,
                combination="x,y",
            )
        if span_extractor_type == "attn":
            self._endpoint_span_extractor = SelfAttentiveSpanExtractor(
                token_embed_dim
            )
        if span_extractor_type == "max_pool":
            self._endpoint_span_extractor = MaxPoolSpanExtractor(
                token_embed_dim
            )
        if span_extractor_type == "mean_pool":
            self._endpoint_span_extractor = MeanPoolSpanExtractor(
                token_embed_dim
            )
        if span_extractor_type == "bi_affine":
            assert self._endpoint_span_extractor.get_output_dim() == token_embed_dim * 2
            self._endpoint_span_extractor = BiAffineSpanExtractor(
                endpoint_extractor=self._endpoint_span_extractor,
                input_size=token_embed_dim,
                project_size=200,
                output_size=200,
            )


        self._visualize_outputs = []

        ####################

        # Set parameters.
        self.use_dep = use_dep
        self._embedder = embedder
        self._loss_weights = loss_weights
        self._max_span_width = max_span_width
        self._display_metrics = self._get_display_metrics(target_task)
        token_emb_dim = self._embedder.get_output_dim()
        span_emb_dim = self._endpoint_span_extractor.get_output_dim()
        if self.use_span_width:
            self.span_width_embedder = SpanWidthEmbedder(num_width_embeddings=max_span_width)
            span_emb_dim += self.span_width_embedder.dim

        if self.use_filter and self.use_tag_embeds and self.use_tree_embeds:
            self.tag_embedder = TagEmbedder(num_width_embeddings=max_span_width, )
            self.tree_embedder = TreeEmbedder()
            self.filter_dropout = torch.nn.Dropout(0.5)
            self.filter_relu = torch.nn.ReLU()
            # self.filter_mlp = nn.Linear(self.tag_embedder.dim + self.tree_embedder.dim, int((self.tag_embedder.dim + self.tree_embedder.dim)/2))
            self.filter_mlp = nn.Linear(self.tag_embedder.dim + self.tree_embedder.dim,
                                        int((self.tag_embedder.dim + self.tree_embedder.dim) ))
            # span_emb_dim += int((self.tag_embedder.dim + self.tree_embedder.dim) /2)
            span_emb_dim += int((self.tag_embedder.dim + self.tree_embedder.dim) )
        else:
            if self.use_tag_embeds:
                self.tag_embedder = TagEmbedder(num_width_embeddings=max_span_width,)
                span_emb_dim += self.tag_embedder.dim
            if self.use_tree_embeds:
                self.tree_embedder = TreeEmbedder()
                span_emb_dim += self.tree_embedder.dim


        # New
        self._feature_size = feature_size
        ####################

        # Create submodules.
        # Helper function to create feedforward networks.
        def make_feedforward(input_dim):
            return FeedForward(
                input_dim=input_dim,
                num_layers=feedforward_params["num_layers"],
                hidden_dims=feedforward_params["hidden_dims"],
                activations=torch.nn.ReLU(),
                dropout=feedforward_params["dropout"],
            )

        # Submodules

        if self.use_dep:
            self.dep_type_embedding = nn.Embedding(len_dep_type+1, token_embed_dim, padding_idx=0)
            gcn_layer = TypeGraphConvolution(token_embed_dim, token_emb_dim)
            self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(1)])
            self.dep_dropout = torch.nn.Dropout(0.5)
            self.relu = torch.nn.ReLU()
            mlp_layers = [nn.Linear(token_embed_dim, int(token_embed_dim/2)),nn.ReLU(),nn.Dropout(0.5),
                          nn.Linear(int(token_embed_dim / 2), int(token_embed_dim / 4)),nn.ReLU(),nn.Dropout(0.5),
                          nn.Linear(int(token_embed_dim / 4), int(token_embed_dim / 8)),nn.ReLU(),nn.Dropout(0.5),
                          nn.Linear(int(token_embed_dim / 8), int(token_embed_dim / 16)),nn.ReLU(),nn.Dropout(0.5),
                          nn.Linear(int(token_embed_dim / 16), dep_dim),nn.ReLU(),nn.Dropout(0.5)]
            self.out_mlp = nn.Sequential(*mlp_layers)

        self._ner = NERTagger(
            pretrain_transformer_indexer = pretrain_transformer_indexers,
            make_feedforward = make_feedforward,
            span_emb_dim = span_emb_dim,
            # feature_size=feature_size,
            data_set_name = data_set_name,
            **modules.pop("ner"),
        )

        # New
        self.use_ner_embeds = use_ner_embeds
        if self.use_ner_embeds:
            num_ner_labels = sorted(self._ner._n_labels.values())[0]
            self._ner_embedder = torch.nn.Linear(num_ner_labels, feature_size)
            span_emb_dim += feature_size

        if self.use_dep:
            params = dict(
                pretrain_transformer_indexer = pretrain_transformer_indexers,
                make_feedforward=make_feedforward,
                span_emb_dim=span_emb_dim,
                feature_size=feature_size,
                token_embed_dim=token_embed_dim,
                use_dep=True,
                dep_dim = dep_dim,
                data_set_name = data_set_name,
                **modules.pop("relation"),
            )
        else:
            params = dict(
                pretrain_transformer_indexer = pretrain_transformer_indexers,
                make_feedforward=make_feedforward,
                span_emb_dim=span_emb_dim,
                feature_size=feature_size,
                token_embed_dim=token_embed_dim,
                use_dep=False,
                data_set_name = data_set_name,
                **modules.pop("relation"),
            )
        if self.relation_head_type == "proper":
            # from_params 
            self._relation = ProperRelationExtractor(**params)
        else:
            raise ValueError(f"Unknown: {dict(relation_head_type=relation_head_type)}")

        ####################

        # Initialize text embedder and all submodules
        for module in [self._ner, self._relation]:
            module_initializer(module)

        initializer(self)

    @staticmethod
    def _get_display_metrics(target_task):
        """
        The `target` is the name of the task used to make early stopping decisions. Show metrics
        related to this task.
        """
        lookup = {
            "ner": [
                f"MEAN__{name}" for name in ["ner_precision", "ner_recall", "ner_f1"]
            ],
            "relation": [
                f"MEAN__{name}"
                for name in ["relation_precision", "relation_recall", "relation_f1"]
            ],
        }
        if target_task not in lookup:
            raise ValueError(
                f"Invalied value {target_task} has been given as the target task."
            )
        return lookup[target_task]

    @staticmethod
    def _debatch(x):
        # TODO: Get rid of this when I find a better way to do it.
        return x if x is None else x.squeeze(0)

    def text_to_span_embeds(self, text_embeddings: torch.Tensor, spans):
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
        return span_embeddings

    def get_attention(self, val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape
        val_us = torch.unsqueeze(val_out, 2)        
        val_us = torch.repeat_interleave(val_us, max_len, 2)    # [b, seq_len, seq_len, word_embedding_dim]
        val_cat = torch.cat((val_us, dep_embed), -1)            # [b, seq_len, seq_len, word_embedding_dim + dep_embed_dim]
        atten_expand = (val_cat.float() * val_cat.float().transpose(1,2))
        attention_score = torch.sum(atten_expand, dim=-1)
        attention_score = attention_score / feat_dim ** 0.5
        # softmax
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
        sum_attention_score = torch.sum(
            exp_attention_score, dim=-1
        ).unsqueeze(dim=-1).repeat(1, 1, max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score

    def forward(
        self,
        text,
        spans,
        dep_span_children,
        ner_labels=None,
        relation_labels=None,
        dep_type_matrix=None,
        tree_info = None,
        pos_tag = None,
        meta_relation_dicts = None
        # spacy_process_matrix = None,
    ):
        # TODO batch forward
        if relation_labels is not None:
            relation_labels = relation_labels.long()

        # 生成 Bert 词嵌入 (Wrod Representation)
        text_embeddings = self._embedder(text, num_wrapping_dims=0) # (n_sents, max_sentence_length, embedding_dim)
        text_mask = (text['input_ids'] != 0)                        # (n_sents, max_sentence_length)
        sentence_lengths = text_mask.sum(dim=1).long()              # (n_sents)

        # 依存句法表示 S-DSR
        dep_feature_embeddings = None
        if self.use_dep:
            sequence_output = text_embeddings
            dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
            dep_adj_matrix = torch.clamp(dep_type_matrix, 0, 1)
            for _, gcn_layer_module in enumerate(self.gcn_layer):
                attention_score = self.get_attention(sequence_output, dep_type_embedding_outputs, dep_adj_matrix)
                sequence_output = gcn_layer_module(sequence_output, attention_score, dep_type_embedding_outputs)
            dep_feature_embeddings = self.out_mlp(sequence_output)

        # 生成跨度表示 (SpanRepresentation)
        ## 当下标为负数时这个span被视作用于padding
        span_mask = (spans[:, :, 0] >= 0).float()  # (n_sents, max_n_spans)
        spans = F.relu(spans.float()).long()  # (n_sents, max_n_spans, 2)
        span_embeddings = self.text_to_span_embeds(text_embeddings, spans)
        if self.use_span_width:
            span_embeddings = self.span_width_embedder(spans, span_embeddings)  # 宽度嵌入
        
        # S-PR & S_CSR
        if self.use_filter and self.use_tag_embeds and self.use_tree_embeds:
            tag_embeddings = self.tag_embedder(pos_tag)

            from fix.debug import fshape
            fshape(tree_info, spans)

            tree_embeddings = self.tree_embedder(tree_info, spans)
            spcsr_embeddings = torch.cat((tag_embeddings, tree_embeddings), -1)
            
            from fix.debug import fshape
            fshape(tag_embeddings, tree_embeddings, spcsr_embeddings)

            kab_embeddings = self.filter_dropout(self.filter_relu(self.filter_mlp(spcsr_embeddings)))
            span_embeddings = torch.cat((span_embeddings, kab_embeddings), -1)
        else:
            if self.use_tag_embeds:
                tag_embeddings = self.tag_embedder(pos_tag)
                span_embeddings = torch.cat((span_embeddings, tag_embeddings), -1)
            if self.use_tree_embeds:
                tag_embeddings = self.tree_embedder(tree_info, spans)
                span_embeddings = torch.cat((span_embeddings, tag_embeddings), -1)
        span_embeds_b = span_embeddings

        # Make calls out to the modules to get results.
        output_ner = {"loss": 0}
        output_relation = {"loss": 0}

        # Make predictions and compute losses for each module
        if self._loss_weights["ner"] > 0:
            # 术语提取
            output_ner = self._ner(
                spans,
                span_mask,
                span_embeddings,
                sentence_lengths,
                ner_labels,
            )
            ner_scores = output_ner.pop("ner_scores")

        # New
        if self._loss_weights["relation"] > 0:
            if getattr(self._relation, "use_ner_scores_for_prune", False):
                raise ValueError("使用到了应该为 False 的值")
                # self._relation._ner_scores = ner_scores
            self._relation._opinion_scores = output_ner["opinion_scores"]
            self._relation._target_scores = output_ner["target_scores"]
            self._relation._text_mask = text_mask
            self._relation._text_embeds = text_embeddings
            if getattr(self._relation, "use_span_loss_for_pruners", False):
                raise ValueError("使用到了应该为 False 的值")
                # self._relation._ner_labels = ner_labels
            output_relation = self._relation(
                spans,
                span_mask,
                # span_embeddings,
                span_embeds_b,
                sentence_lengths,
                relation_labels,
                dep_feature_embeddings,
                meta_relation_dicts,
                # spacy_process_matrix[0][0],
            )

        # Use `get` since there are some cases where the output dict won't have a loss - for
        # instance, when doing prediction.
        loss = (
            + self._loss_weights["ner"] * output_ner.get("loss", 0)
            + self._loss_weights["relation"] * output_relation.get("loss", 0)
        )

        weight = 1.0
        loss *= torch.tensor(weight)

        output_dict = dict(
            relation=output_relation,
            ner=output_ner,
        )
        output_dict["loss"] = loss

        return output_dict

    def update_span_embeddings(
        self,
        span_embeddings,
        span_mask,
        top_span_embeddings,
        top_span_mask,
        top_span_indices,
    ):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if (
                    top_span_mask[sample_nr, top_span_nr] == 0
                    or span_mask[sample_nr, span_nr] == 0
                ):
                    break
                new_span_embeddings[sample_nr, span_nr] = top_span_embeddings[
                    sample_nr, top_span_nr
                ]
        return new_span_embeddings

     
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.
        """

        doc = copy.deepcopy(output_dict["metadata"])

        if self._loss_weights["ner"] > 0:
            for predictions, sentence in zip(output_dict["ner"]["predictions"], doc):
                sentence.predicted_ner = predictions

        if self._loss_weights["relation"] > 0:
            for predictions, sentence in zip(
                output_dict["relation"]["predictions"], doc
            ):
                sentence.predicted_relations = predictions

        return doc

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (
            list(metrics_ner.keys())
            + list(metrics_relation.keys())
        )
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
            list(metrics_ner.items())
            + list(metrics_relation.items())
        )

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
