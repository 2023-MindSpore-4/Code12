import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import msadapter.pytorch.nn.functional as F

from ms_allennlp_fix.type_define import Vocabulary, RegularizerApplicator, Model
from ms_allennlp_fix.modules import TimeDistributed
import ms_allennlp_fix.util as util
from ms_allennlp_fix.token_indexer_type import (
    PreTrainTransformerTokenIndexer
)

from span_model.models.shared import BiAffine, SpanLengthCrossEntropy, BagPairScorer, BiAffineV2
from span_model.training.relation_metrics import RelationMetrics
from span_model.models.entity_beam_pruner import Pruner
import span_model.dataset_readers.document as document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import json
from pydantic import BaseModel
import networkx as nx
import spacy
import nltk
from span_model.models.shared import FocalLoss
from span_model.models.SelfAttention import SelfAttention


class PruneOutput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    span_embeddings: torch.Tensor
    span_mention_scores: torch.Tensor
    num_spans_to_keep: torch.Tensor
    span_mask: torch.Tensor
    span_indices: torch.Tensor
    spans: torch.Tensor


def analyze_info(info: dict):
    for k, v in info.items():
        if isinstance(v, torch.Size):
            v = tuple(v)
        info[k] = str(v)
    logging.info(json.dumps(info, indent=2))


class DistanceEmbedder(torch.nn.Module):
    def __init__(self, dim=128, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedder = torch.nn.Embedding(self.vocab_size, self.dim)

    def to_distance_buckets(self, spans_a: torch.Tensor, spans_b: torch.Tensor) -> torch.Tensor:
        bs, num_a, dim = spans_a.shape
        bs, num_b, dim = spans_b.shape
        assert dim == 2

        spans_a = spans_a.view(bs, num_a, 1, dim)
        spans_b = spans_b.view(bs, 1, num_b, dim)
        k = spans_b[..., 0]
        g= spans_a[..., 1]
        d_ab = torch.abs(spans_b[..., 0] - spans_a[..., 1])
        d_ba = torch.abs(spans_a[..., 0] - spans_b[..., 1])
        distances = torch.minimum(d_ab, d_ba)

        # pos_a = spans_a.float().mean(dim=-1).unsqueeze(dim=-1)  # bs, num_spans, 1
        # pos_b = spans_b.float().mean(dim=-1).unsqueeze(dim=-2)  # bs, 1, num_spans
        # distances = torch.abs(pos_a - pos_b)

        x = util.bucket_values(distances, num_total_buckets=self.vocab_size)
        # [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]
        x = x.long()
        assert x.shape == (bs, num_a, num_b)
        return x

    def forward(self, spans_a: torch.Tensor, spans_b: torch.Tensor, matrix_dist) -> torch.Tensor:
        buckets = self.to_distance_buckets(spans_a, spans_b)
        # buckets = util.bucket_values(matrix_dist.reshape(1,matrix_dist.shape[0],matrix_dist.shape[1]), num_total_buckets=self.vocab_size)
        buckets = buckets.long()
        x = self.embedder(buckets)  # bs, num_spans, num_spans, dim
        return x


def global_max_pool1d(x: torch.Tensor) -> torch.Tensor:
    bs, seq_len, features = x.shape
    x = x.transpose(-1, -2)
    x = F.adaptive_max_pool1d(x, output_size=1, return_indices=False)
    x = x.transpose(-1, -2)
    x = x.squeeze(dim=1)
    assert tuple(x.shape) == (bs, features)
    return x


def test_pool():
    x = torch.zeros(3, 100, 32)
    y = global_max_pool1d(x)
    print(dict(x=x.shape, y=y.shape))


class ProperRelationExtractor(Model):
    def __init__(
        self,
        pretrain_transformer_indexer: PreTrainTransformerTokenIndexer,  
        make_feedforward: Callable,
        span_emb_dim: int,
        feature_size: int,
        spans_per_word: float,
        positive_label_weight: float = 1.0,
        regularizer: Optional[RegularizerApplicator] = None,
        use_distance_embeds: bool = False,
        use_pair_feature_maxpool: bool = False,
        use_pair_feature_cls: bool = False,
        use_bi_affine_classifier: bool = False,
        neg_class_weight: float = -1,
        span_length_loss_weight_gamma: float = 0.0,
        use_bag_pair_scorer: bool = False,
        use_bi_affine_v2: bool = False,
        use_pruning: bool = True,
        use_single_pool: bool = False,
        token_embed_dim: int = 768,
        use_dep=True,
        dep_dim=128,
        is_attention=True,
        is_lock=False,
        data_set_name: str = None,
        **kwargs,  # noqa
    ) -> None:
        super().__init__()
        self.pretrain_transformer_indexer = pretrain_transformer_indexer
        self.regularizer = regularizer

        # print(dict(unused_keys=kwargs.keys()))
        # print(dict(locals=locals()))
        self.use_single_pool = use_single_pool
        self.use_pruning = use_pruning
        self.use_bi_affine_v2 = use_bi_affine_v2
        self.use_bag_pair_scorer = use_bag_pair_scorer
        self.span_length_loss_weight_gamma = span_length_loss_weight_gamma
        self.use_bi_affine_classifier = use_bi_affine_classifier
        self.use_distance_embeds = use_distance_embeds
        self.use_pair_feature_maxpool = use_pair_feature_maxpool
        self.use_pair_feature_cls = use_pair_feature_cls
        self._text_embeds: Optional[torch.Tensor] = None
        self._text_mask: Optional[torch.Tensor] = None
        self._spans_a: Optional[torch.Tensor] = None
        self._spans_b: Optional[torch.Tensor] = None
        self.use_dep = use_dep
        self.is_lock = is_lock
        token_emb_dim = token_embed_dim
        relation_scorer_dim = 2 * span_emb_dim
        if self.is_lock:
            self.expand_mlp_a = nn.Sequential(nn.Linear(dep_dim,span_emb_dim))
            self.expand_mlp_b = nn.Sequential(nn.Linear(dep_dim, span_emb_dim))
            self.dep_mlp_a = nn.Sequential(nn.Linear(span_emb_dim*2,span_emb_dim))
            self.dep_mlp_b = nn.Sequential(nn.Linear(span_emb_dim * 2, span_emb_dim))
        if self.use_distance_embeds:
            self.d_embedder = DistanceEmbedder()
            relation_scorer_dim += self.d_embedder.dim
        if self.use_pair_feature_maxpool:
            relation_scorer_dim += token_emb_dim
        if self.use_pair_feature_cls:
            relation_scorer_dim += token_emb_dim
        if self.use_dep:
            relation_scorer_dim += 2*dep_dim

        print(dict(token_emb_dim=token_emb_dim, span_emb_dim=span_emb_dim, relation_scorer_dim=relation_scorer_dim))
        self._namespaces = [
            ns for ns in pretrain_transformer_indexer.get_extend_name_spaces() 
            if 'relation_labels' in ns
            ]
        self._n_labels = {
            name: pretrain_transformer_indexer.get_voc_size(name)
              for name in self._namespaces }
        assert len(self._n_labels) == 1, f"获得了不正确数量的字典 当前为{len(self._n_labels)} != 1"
        n_labels = list(self._n_labels.values())[0] + 1
        if self.use_bi_affine_classifier:
            self._bi_affine_classifier = BiAffine(span_emb_dim, project_size=200, output_size=n_labels)
        if self.use_bi_affine_v2:
            self._bi_affine_v2 = BiAffineV2(span_emb_dim, project_size=200, output_size=n_labels)

        self._mention_pruners = torch.nn.ModuleDict()
        self._relation_feedforwards = torch.nn.ModuleDict()
        self._relation_scorers = torch.nn.ModuleDict()
        self._relation_metrics = {}
        self.is_attention = is_attention
        if self.is_attention:
            self.attention = SelfAttention(dep_dim,2,0.2)
            # self.attention = SelfAttention(token_emb_dim, 2, 0.2)

        self._pruner_o = self._make_pruner(span_emb_dim, make_feedforward)
        self._pruner_t = self._make_pruner(span_emb_dim, make_feedforward)
        if not self.use_pruning:
            self._pruner_o, self._pruner_t = None, None
        if self.use_single_pool:
            assert self.use_pruning
            self._pruner_o = self._pruner_t

        for namespace in self._namespaces:
            relation_feedforward = make_feedforward(input_dim=relation_scorer_dim)
            if self.use_bag_pair_scorer:
                relation_feedforward = BagPairScorer(make_feedforward, span_emb_dim)
            self._relation_feedforwards[namespace] = relation_feedforward
            relation_scorer = torch.nn.Linear(
                relation_feedforward.get_output_dim(), self._n_labels[namespace] + 1
            )
            self._relation_scorers[namespace] = relation_scorer

            self._relation_metrics[namespace] = RelationMetrics()

        self._spans_per_word = spans_per_word

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        if self.span_length_loss_weight_gamma != 0:
            assert neg_class_weight == -1
            self._loss = SpanLengthCrossEntropy(
                gamma=self.span_length_loss_weight_gamma, reduction="sum", ignore_index=-1)

        if neg_class_weight != -1:
            assert len(self._namespaces) == 1
            num_pos_classes = self._n_labels[self._namespaces[0]]
            weight = torch.tensor([neg_class_weight] + [1.0] * num_pos_classes)
            print(dict(relation_neg_class_weight=weight))
            self._loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1, weight=weight)

        # todo 使用focal loss
        self._loss = FocalLoss(gamma=2,  ignore_index=-1, reduction="sum")

        # print(dict(relation_loss_fn=self._loss))
        self._active_namespace = f"{data_set_name}__relation_labels"

    def _make_pruner(self, span_emb_dim:int, make_feedforward:Callable):
        mention_feedforward = make_feedforward(input_dim=span_emb_dim)

        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(
                torch.nn.Linear(mention_feedforward.get_output_dim(), 1)
            ),
        )
        return Pruner(feedforward_scorer, use_external_score=True)

     
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,
        span_mask,
        span_embeddings,  # TODO: add type.
        sentence_lengths,
        relation_labels: torch.IntTensor = None,
        dep_feature_embeddings = None,
        metadata_relation_dicts: List[Dict[Tuple[int, int], Any]] = None,
        # spacy_process_matrix = None,
    ) -> Dict[str, torch.Tensor]:
        pruned_o: PruneOutput = self._prune_spans(spans, span_mask, span_embeddings, sentence_lengths, "opinion")
        pruned_t: PruneOutput = self._prune_spans(spans, span_mask, span_embeddings, sentence_lengths, "target")
        # matrix_filter, matrix_dist = spacy_pruned(pruned_o, pruned_t, raw_sentences)
        # pruned_o,pruned_t = tag_pruned(pruned_o, pruned_t, raw_sentences)
        matrix_filter, matrix_dist = None, None
        relation_scores = self._compute_relation_scores(pruned_o, pruned_t,matrix_filter,matrix_dist,dep_feature_embeddings)

        prediction_dict, predictions = self.predict(
            spans_a=pruned_o.spans.detach(),
            spans_b=pruned_t.spans.detach(),
            relation_scores=relation_scores.detach(),
            num_keep_a=pruned_o.num_spans_to_keep.detach(),
            num_keep_b=pruned_t.num_spans_to_keep.detach(),
        )

        output_dict = {"predictions": predictions}

        # Evaluate loss and F1 if labels were provided.
        if relation_labels is not None:
            # Compute cross-entropy loss.
            gold_relations = self._get_pruned_gold_relations(
                relation_labels, pruned_o, pruned_t
            )

            self._relation_scores, self._gold_relations = relation_scores, gold_relations
            cross_entropy = self._get_cross_entropy_loss(
                relation_scores, gold_relations
            )

            # Compute F1.
            relation_metrics = self._relation_metrics[self._active_namespace]
            relation_metrics(prediction_dict, metadata_relation_dicts)

            output_dict["loss"] = cross_entropy
        return output_dict

    def _prune_spans(self, spans, span_mask, span_embeddings, sentence_lengths, name: str) -> PruneOutput:
        if not self.use_pruning:
            bs, num_spans, dim = span_embeddings.shape
            device = span_embeddings.device
            return PruneOutput(
                spans=spans,
                span_mask=span_mask.unsqueeze(dim=-1),
                span_embeddings=span_embeddings,
                num_spans_to_keep=torch.full((bs,), fill_value=num_spans, device=device, dtype=torch.long),
                span_indices=torch.arange(num_spans, device=device, dtype=torch.long).view(1, num_spans).expand(bs, -1),
                span_mention_scores=torch.zeros(bs, num_spans, 1, device=device),
            )

        pruner = dict(opinion=self._pruner_o, target=self._pruner_t)[name]
        if self.use_single_pool:
            self._opinion_scores = torch.maximum(self._opinion_scores, self._target_scores)
            self._target_scores = self._opinion_scores
        mention_scores = dict(opinion=self._opinion_scores, target=self._target_scores)[name]
        pruner.set_external_score(mention_scores.detach())

        # Prune
        num_spans = spans.size(1)  # Max number of spans for the minibatch.

        # Keep different number of spans for each minibatch entry.
        num_spans_to_keep = torch.ceil(
            sentence_lengths.float() * self._spans_per_word
        ).long()

        # if num_spans_to_keep.item() > 15:
        #     num_spans_to_keep[0] = 15

        outputs = pruner(span_embeddings, span_mask, num_spans_to_keep, spans=spans, name=name)
        (
            top_span_embeddings,
            top_span_mask,
            top_span_indices,
            top_span_mention_scores,
            num_spans_kept,
        ) = outputs

        top_span_mask = top_span_mask.unsqueeze(-1)

        flat_top_span_indices = util.flatten_and_batch_shift_indices(
            top_span_indices, num_spans
        )
        top_spans = util.batched_index_select(
            spans, top_span_indices, flat_top_span_indices
        )

        return PruneOutput(
            span_embeddings=top_span_embeddings,
            span_mention_scores=top_span_mention_scores,
            num_spans_to_keep=num_spans_to_keep,
            span_mask=top_span_mask,
            span_indices=top_span_indices,
            spans=top_spans,
        )

    def predict(self, spans_a, spans_b, relation_scores, num_keep_a, num_keep_b):
        preds_dict = []
        predictions = []
        for i in range(relation_scores.shape[0]):
            # Each entry/sentence in batch
            pred_dict_sent, predictions_sent = self._predict_sentence(
                spans_a[i], spans_b[i], relation_scores[i],
                num_keep_a[i], num_keep_b[i]
            )
            preds_dict.append(pred_dict_sent)
            predictions.append(predictions_sent)

        return preds_dict, predictions

    def _predict_sentence(
        self, top_spans_a, top_spans_b, relation_scores, num_keep_a, num_keep_b
    ):
        num_a = num_keep_a.item()  # noqa
        num_b = num_keep_b.item()  # noqa
        spans_a = [tuple(x) for x in top_spans_a.tolist()]
        spans_b = [tuple(x) for x in top_spans_b.tolist()]

        # Iterate over all span pairs and labels. Record the span if the label isn't null.
        predicted_scores_raw, predicted_labels = relation_scores.max(dim=-1)
        softmax_scores = F.softmax(relation_scores, dim=-1)
        predicted_scores_softmax, _ = softmax_scores.max(dim=-1)
        predicted_labels -= 1  # Subtract 1 so that null labels get -1.

        ix = (predicted_labels >= 0)  # TODO: Figure out their keep_mask (relation.py:202)

        res_dict = {}
        predictions = []

        for i, j in ix.nonzero(as_tuple=False):
            span_1 = spans_a[i]
            span_2 = spans_b[j]
            label = predicted_labels[i, j].item()
            raw_score = predicted_scores_raw[i, j].item()
            softmax_score = predicted_scores_softmax[i, j].item()

            label_name = self.pretrain_transformer_indexer.get_str_of_idx(
                label, self._active_namespace
            )
            res_dict[(span_1, span_2)] = label
            list_entry = (
                span_1[0],
                span_1[1],
                span_2[0],
                span_2[1],
                label_name,
                raw_score,
                softmax_score,
            )
            predictions.append(
                list_entry
                # document.PredictedRelation(list_entry, sentence, sentence_offsets=True)
            )

        return res_dict, predictions

    # TODO: This code is repeated elsewhere. Refactor.
     
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._relation_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            to_update = {
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
                f"{prefix}_f1": f1,
            }
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__relation_{name}"] = (
                sum(values) / len(values) if values else 0
            )
            res.update(res_avg)

        return res

    def _make_pair_features(
            self, a: torch.Tensor, b: torch.Tensor, 
            matrix_dist, dep_feature_embeddings,
            a_span:torch.Tensor, b_span:torch.Tensor
        ) -> torch.Tensor:
        assert a.shape == b.shape
        bs, num_a, num_b, size = a.shape
        features = [a,]

        if self.use_dep:
            def compute_dep_embeddings(x_span:torch.Tensor):
                x_dep_embeddings = []
                for i in range(x_span.shape[1]):
                    x_start = x_span[0, i, 0].item()
                    x_end = x_span[0, i, 1].item() + 1
                    x_embeddings = dep_feature_embeddings[:, x_start:x_end, :]
                    x_mask = torch.ones(x_embeddings.shape[0], x_embeddings.shape[1])
                    process_x_dep_embed = torch.sum(self.attention(x_embeddings, x_mask), dim=1).unsqueeze(0)
                    x_dep_embeddings.append(process_x_dep_embed)
                x_final_dep_embed = torch.cat(x_dep_embeddings, dim=1)
                return x_final_dep_embed
            
            a_final_dep_embed = compute_dep_embeddings(a_span)
            b_final_dep_embed = compute_dep_embeddings(b_span)

            a_final_dep_embed:torch.Tensor
            a_final_dep_embed = a_final_dep_embed.unsqueeze(2)
            a_new_span_embed = a_final_dep_embed.expand(-1, -1, a_final_dep_embed.shape[1], -1)

            b_final_dep_embed:torch.Tensor
            b_final_dep_embed = b_final_dep_embed.unsqueeze(1)
            b_new_span_embed = b_final_dep_embed.expand(-1, b_final_dep_embed.shape[2], -1, -1)
            # a_b_product_embed = a_new_span_embed * b_new_span_embed
            features.append(a_new_span_embed)
            features.append(b)
            features.append(b_new_span_embed)
        else:
            features.append(b)

        if self.use_pair_feature_maxpool:
            x = self._text_embeds
            c = global_max_pool1d(x)  # [bs, size]
            bs, size = c.shape
            c = c.view(bs, 1, 1, size).expand(-1, num_a, num_b, -1)
            features.append(c)

        if self.use_pair_feature_cls:
            c = self._text_embeds[:, 0, :]
            bs, size = c.shape
            c = c.view(bs, 1, 1, size).expand(-1, num_a, num_b, -1)
            features.append(c)

        if self.use_distance_embeds:
            features.append(self.d_embedder(self._spans_a, self._spans_b, matrix_dist))

        x = torch.cat(features, dim=-1)
        return x

    def _compute_span_pair_embeddings(self, a: torch.Tensor, b: torch.Tensor, matrix_dist,dep_feature_embeddings,a_span,b_span) -> torch.Tensor:
        c = self._make_pair_features(a, b, matrix_dist,dep_feature_embeddings,a_span,b_span)
        if self.use_bi_affine_classifier:
            c = self._bi_affine_classifier(a, b)
        return c

    def _compute_relation_scores(self, pruned_a: PruneOutput, pruned_b: PruneOutput,matrix_filter,matrix_dist,dep_feature_embeddings):

        # debuging!
        from fix.debug import fshape

        if self.span_length_loss_weight_gamma != 0:
            bs, num_a, _ = pruned_a.spans.shape
            bs, num_b, _ = pruned_b.spans.shape
            widths_a = pruned_a.spans[..., [1]] - pruned_a.spans[..., [0]] + 1
            widths_b = pruned_b.spans[..., [1]] - pruned_b.spans[... ,[0]] + 1
            widths_a = widths_a.view(bs, num_a, 1, 1)
            widths_b = widths_b.view(bs, 1, num_b, 1)
            widths = (widths_a + widths_b) / 2

            fshape(widths_a, widths_b)

            # todo 暂时尝试下focal loss
            # self._loss.lengths = widths.view(bs * num_a * num_b)
        a_span,b_span = pruned_a.spans, pruned_b.spans
        a_orig, b_orig = pruned_a.span_embeddings, pruned_b.span_embeddings

        fshape(a_span, b_span, a_orig, b_orig)  # TODO 在云端除错

        bs, num_a, size = a_orig.shape
        bs, num_b, size = b_orig.shape
        chunk_size = max(10000 // num_a, 1)
        # logging.info(dict(a=num_a, b=num_b, chunk_size=chunk_size))
        pool = []

        for i in range(0, num_a, chunk_size):
            a = a_orig[:, i:i + chunk_size, :]
            num_chunk = a.shape[1]
            a = a.view(bs, num_chunk, 1, size).expand(-1, -1, num_b, -1)
            b = b_orig.view(bs, 1, num_b, size).expand(-1, num_chunk, -1, -1)

            fshape(a, b)

            assert a.shape == b.shape
            self._spans_a = pruned_a.spans[:, i:i + chunk_size, :]
            self._spans_b = pruned_b.spans

            fshape(self._spans_a, self._spans_b)

            embeds = self._compute_span_pair_embeddings(a, b,matrix_dist,dep_feature_embeddings,a_span,b_span)
            self._relation_embeds = embeds

            fshape(embeds)  # there

            if self.use_bi_affine_classifier:
                scores = embeds
            else:
                relation_feedforward = self._relation_feedforwards[self._active_namespace]
                relation_scorer = self._relation_scorers[self._active_namespace]

                fshape(embeds)

                embeds = torch.flatten(embeds, end_dim=-2)
                projected = relation_feedforward(embeds)
                scores = relation_scorer(projected)

            scores = scores.view(bs, num_chunk, num_b, -1)
            
            fshape(scores)

            if self.use_bi_affine_v2:
                scores += self._bi_affine_v2(a, b)
            pool.append(scores)
        scores = torch.cat(pool, dim=1)
            
        fshape(scores)

        if torch.isnan(scores[0][0][0][0] ):
            cc= np.array(embeds.detach())
            print(888)
            # NOTE 这里像debug用的代码，而没有及时删除
        # scores = spacy_process_scores(scores, matrix_filter)
        return scores

    @staticmethod
    def _get_pruned_gold_relations(relation_labels: torch.Tensor, pruned_a: PruneOutput, pruned_b: PruneOutput) -> torch.Tensor:
        """
        Loop over each slice and get the labels for the spans from that slice.
        All labels are offset by 1 so that the "null" label gets class zero. This is the desired
        behavior for the softmax. Labels corresponding to masked relations keep the label -1, which
        the softmax loss ignores.
        """
        # TODO: Test and possibly optimize.
        relations = []

        indices_a, masks_a = pruned_a.span_indices, pruned_a.span_mask.bool()
        indices_b, masks_b = pruned_b.span_indices, pruned_b.span_mask.bool()

        for i in range(relation_labels.shape[0]):
            # Each entry in batch
            entry = relation_labels[i]
            entry = entry[indices_a[i], :][:, indices_b[i]]
            mask_entry = masks_a[i] & masks_b[i].transpose(0, 1)
            assert entry.shape == mask_entry.shape
            entry[mask_entry] += 1
            entry[~mask_entry] = -1
            relations.append(entry)

        # return torch.cat(relations, dim=0)
        # This should be a mistake, don't want to concat items within a batch together
        # Likely undiscovered because current bs=1 and _get_loss flattens everything
        return torch.stack(relations, dim=0)

    def _get_cross_entropy_loss(self, relation_scores, relation_labels):
        """
        Compute cross-entropy loss on relation labels. Ignore diagonal entries and entries giving
        relations between masked out spans.
        """
        # Need to add one for the null class.
        n_labels = self._n_labels[self._active_namespace] + 1
        scores_flat = relation_scores.view(-1, n_labels)
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = relation_labels.view(-1)
        # Compute cross-entropy loss.
        loss = self._loss(scores_flat, labels_flat)
        if torch.isnan(loss):
            print(888)
        return loss

def spacy_process_scores(scores,matrix_filter):
    for i in range(0,matrix_filter.shape[0]):
        for j in range(0,matrix_filter.shape[1]):
            jug = matrix_filter[i][j]
            if jug == 0:
                scores[0][j][i][0] = 1
                scores[0][j][i][1] = 0
                scores[0][j][i][2] = 0
                scores[0][j][i][3] = 0
    return scores

def spacy_pruned(pruned_o,pruned_t,raw_sentences):
    opinion_spans = pruned_o.spans
    target_spans = pruned_t.spans
    matrix_dist, matrix_filter = spacy_process(raw_sentences, target_spans[0].tolist(), opinion_spans[0].tolist(), 5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matrix_dist = torch.FloatTensor(matrix_dist)
    matrix_dist = matrix_dist.to(device)
    matrix_filter = torch.LongTensor(matrix_filter)
    matrix_filter = matrix_filter.to(device)
    return matrix_filter, matrix_dist


def spacy_process(sentence_list, aspect_term_ids, opinion_term_ids, threshold):
    matrix_dist = [[999] * len(opinion_term_ids) for _ in range(len(aspect_term_ids))]
    matrix_filter = [[1] * len(opinion_term_ids) for _ in range(len(aspect_term_ids))]


    sentence = ' '.join(sentence_list[0:])

    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sentence)
    except NameError as e:
        raise RuntimeError('Fail to load nlp model, maybe you forget to download en_core_web_sm')

    sentence2 = ''
    for token in doc:
        sentence2 = sentence2 + str(token) + ' '
    sentence2 = sentence2.strip()

    aspect_terms = []
    for m in range(len(aspect_term_ids)):
        aspect_term = ' '.join(sentence_list[aspect_term_ids[m][0]:aspect_term_ids[m][1] + 1])
        aspect_term_temp = ''
        doc = nlp(aspect_term)
        for token in doc:
            aspect_term_temp = aspect_term_temp + str(token) + ' '
        aspect_term_temp = aspect_term_temp.strip()
        aspect_terms.append(aspect_term_temp)

    opinion_terms = []
    for m in range(len(opinion_term_ids)):
        opinion_term = ' '.join(sentence_list[opinion_term_ids[m][0]:opinion_term_ids[m][1] + 1])
        opinion_term_temp = ''
        doc = nlp(opinion_term)
        for token in doc:
            opinion_term_temp = opinion_term_temp + str(token) + ' '
        opinion_term_temp = opinion_term_temp.strip()
        opinion_terms.append(opinion_term_temp)

    doc = nlp(sentence2)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{}_{}'.format(token.lower_, token.i),
                          '{}_{}'.format(child.lower_, child.i)))
    graph = nx.Graph(edges)

    for m in range(len(aspect_terms)):
        for n in range(len(opinion_terms)):
            aspect_term = aspect_terms[m]
            opinion_term = opinion_terms[n]
            aspects = [a.lower() for a in aspect_term.split()]
            opinions = [a.lower() for a in opinion_term.split()]

            # Load spacy's dependency tree into a networkx graph

            cnt_aspect = 0
            cnt_opinion = 0
            aspect_ids = [0] * len(aspects)
            opinion_ids = [0] * len(opinions)
            for token in doc:
                # Record the position of aspect terms
                if cnt_aspect < len(aspects) and token.lower_ == aspects[cnt_aspect]:
                    aspect_ids[cnt_aspect] = token.i

                    cnt_aspect += 1
                # Record the position of opinion terms
                if cnt_opinion < len(opinions) and token.lower_ == opinions[cnt_opinion]:
                    opinion_ids[cnt_opinion] = token.i
                    cnt_opinion += 1

            dist = [0.0] * len(doc)
            for i, word in enumerate(doc):
                source = '{}_{}'.format(word.lower_, word.i)
                sum = 0
                max_dist = 0
                for aspect_id, aspect in zip(aspect_ids, aspects):
                    target = '{}_{}'.format(aspect, aspect_id)
                    try:
                        sum += nx.shortest_path_length(graph, source=source, target=target)
                    except:
                        sum += len(doc)  # No connection between source and target
                        flag = 0
                dist[i] = sum / len(aspects)

            aspect_opinion_dist = 0

            for i in range(len(opinions)):
                aspect_opinion_dist += dist[opinion_ids[i]]
            if len(opinions) == 0:
                continue

            aspect_opinion_dist = aspect_opinion_dist / len(opinions)
            matrix_dist[m][n] = aspect_opinion_dist
            if aspect_opinion_dist > threshold:
                matrix_filter[m][n] = 0
    return matrix_dist, matrix_filter