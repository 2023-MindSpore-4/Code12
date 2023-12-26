"""
This is basically a copy of AllenNLP's Pruner module, but with support for entity beams.
"""

from typing import Tuple, Union
import nltk

import msadapter.pytorch as torch

import ms_allennlp_fix.util as util
from ms_allennlp_fix.modules import TimeDistributed

def make_pruner(scorer, entity_beam=False, gold_beam=False):
    """
    Create a pruner that either takes outputs of other scorers (i.e. entity beam), or uses its own
    scorer (the `default_scorer`).
    """
    item_scorer = torch.nn.Sequential(
        TimeDistributed(scorer),
        TimeDistributed(torch.nn.Linear(scorer.get_output_dim(), 1)),
    )
    min_score_to_keep = 1e-10 if entity_beam else None

    return Pruner(item_scorer, entity_beam, gold_beam, min_score_to_keep)


class Pruner(torch.nn.Module):
    """
    参数化评分并根据阈值进行修剪

    Parameters
    ----------
    scorer : ``torch.nn.Module``, required.
        A module which, given a tensor of shape (batch_size, num_items, embedding_size),
        produces a tensor of shape (batch_size, num_items, 1), representing a scalar score
        per item in the tensor.
    entity_beam: bool, optional.
        If True, use class scores output from another module instead of using own scorer.
    gold_beam: bool, optional.
       If True, use gold arguments.
    min_score_to_keep : float, optional.
        If given, only keep items that score at least this high.
    """

    def __init__(
        self,
        scorer: torch.nn.Module,
        entity_beam: bool = False,
        gold_beam: bool = False,
        min_score_to_keep: float = None,
        use_external_score: bool = False,
    ) -> None:
        super().__init__()
        # If gold beam is on, then entity beam must be off and min_score_to_keep must be None.
        assert not (gold_beam and ((min_score_to_keep is not None) or entity_beam))
        self._scorer = scorer
        self._entity_beam = entity_beam
        self._gold_beam = gold_beam
        self._min_score_to_keep = min_score_to_keep
        self._use_external_score = use_external_score
        if self._use_external_score:
            self._scorer = None
        self._scores = None

    def set_external_score(self, x: torch.Tensor):
        self._scores = x

     
    def forward(
        self,  # pylint: disable=arguments-differ
        embeddings: torch.FloatTensor,
        mask: torch.LongTensor,
        num_items_to_keep: Union[int, torch.LongTensor],
        class_scores: torch.FloatTensor = None,
        gold_labels: torch.long = None,
        extra_scores: torch.FloatTensor = None,  # Scores to add to scorer output
        raw_sentences = None,
        spans = None,
        name = None
    ) -> Tuple[
        torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor
    ]:
        """
        Extracts the top-k scoring items with respect to the scorer. We additionally return
        the indices of the top-k in their original order, not ordered by score, so that downstream
        components can rely on the original ordering (e.g., for knowing what spans are valid
        antecedents in a coreference resolution model). May use the same k for all sentences in
        minibatch, or different k for each.

        Parameters
        ----------
        embeddings : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_items, embedding_size), containing an embedding for
            each item in the list that we want to prune.
        mask : ``torch.LongTensor``, required.
            A tensor of shape (batch_size, num_items), denoting unpadded elements of
            ``embeddings``.
        num_items_to_keep : ``Union[int, torch.LongTensor]``, required.
            If a tensor of shape (batch_size), specifies the number of items to keep for each
            individual sentence in minibatch.
            If an int, keep the same number of items for all sentences.
        class_scores:
           Class scores to be used with entity beam.
        candidate_labels: If in debugging mode, use gold labels to get beam.

        Returns
        -------
        top_embeddings : ``torch.FloatTensor``
            The representations of the top-k scoring items.
            Has shape (batch_size, max_num_items_to_keep, embedding_size).
        top_mask : ``torch.LongTensor``
            The corresponding mask for ``top_embeddings``.
            Has shape (batch_size, max_num_items_to_keep).
        top_indices : ``torch.IntTensor``
            The indices of the top-k scoring items into the original ``embeddings``
            tensor. This is returned because it can be useful to retain pointers to
            the original items, if each item is being scored by multiple distinct
            scorers, for instance. Has shape (batch_size, max_num_items_to_keep).
        top_item_scores : ``torch.FloatTensor``
            The values of the top-k scoring items.
            Has shape (batch_size, max_num_items_to_keep, 1).
        num_items_kept
        """
        # If an int was given for number of items to keep, construct tensor by repeating the value.
        if isinstance(num_items_to_keep, int):
            batch_size = mask.size(0)
            # Put the tensor on same device as the mask.
            num_items_to_keep = num_items_to_keep * torch.ones(
                [batch_size], dtype=torch.long, device=mask.device
            )

        mask = mask.unsqueeze(-1)
        num_items = embeddings.size(1)
        num_items_to_keep = torch.clamp(num_items_to_keep, max=num_items)

        # Shape: (batch_size, num_items, 1)
        # If entity beam is one, use the class scores. Else ignore them and use the scorer.
        if self._entity_beam:
            scores, _ = class_scores.max(dim=-1)
            scores = scores.unsqueeze(-1)
        # If gold beam is one, give a score of 0 wherever the gold label is non-zero (indicating a
        # non-null label), otherwise give a large negative number.
        elif self._gold_beam:
            scores = torch.where(
                gold_labels > 0,
                torch.zeros_like(gold_labels, dtype=torch.float),
                -1e20 * torch.ones_like(gold_labels, dtype=torch.float),
            )
            scores = scores.unsqueeze(-1)
        else:
            if self._use_external_score:
                scores = self._scores
            else:
                scores = self._scorer(embeddings)
            if extra_scores is not None:
                # Assumes extra_scores is already in [0, 1] range
                scores = scores.sigmoid() + extra_scores

        # If we're only keeping items that score above a given threshold, change the number of kept
        # items here.
        if self._min_score_to_keep is not None:
            num_good_items = torch.sum(
                scores > self._min_score_to_keep, dim=1
            ).squeeze()
            num_items_to_keep = torch.min(num_items_to_keep, num_good_items)
        # If gold beam is on, keep the gold items.
        if self._gold_beam:
            num_items_to_keep = torch.sum(gold_labels > 0, dim=1)

        # Always keep at least one item to avoid edge case with empty matrix.
        max_items_to_keep = max(num_items_to_keep.max().item(), 1)

        if scores.size(-1) != 1 or scores.dim() != 3:
            raise ValueError(
                f"The scorer passed to Pruner must produce a tensor of shape"
                f"(batch_size, num_items, 1), but found shape {scores.size()}"
            )
        # Make sure that we don't select any masked items by setting their scores to be very
        # negative.  These are logits, typically, so -1e20 should be plenty negative.
        # NOTE(`mask` needs to be a byte tensor now.)
        scores = util.replace_masked_values(scores, mask.bool(), -1e20)


        # Shape: (batch_size, max_num_items_to_keep, 1)
        _, top_indices = scores.topk(max_items_to_keep, 1)

        # Mask based on number of items to keep for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices_mask = util.get_mask_from_sequence_lengths(
            num_items_to_keep, max_items_to_keep
        )
        top_indices_mask = top_indices_mask.bool()

        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = top_indices.squeeze(-1)

        # Fill all masked indices with largest "top" index for that sentence, so that all masked
        # indices will be sorted to the end.
        # Shape: (batch_size, 1)
        fill_value, _ = torch.max(top_indices, dim=1)
        fill_value = fill_value.unsqueeze(-1)
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = torch.where(top_indices_mask, top_indices, fill_value)

        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``embeddings`` tensor).
        top_indices, _ = torch.sort(top_indices, 1)

        # Shape: (batch_size * max_num_items_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select items for each element in the batch.
        flat_top_indices = util.flatten_and_batch_shift_indices(top_indices, num_items)

        # Shape: (batch_size, max_num_items_to_keep, embedding_size)
        top_embeddings = util.batched_index_select(
            embeddings, top_indices, flat_top_indices
        )

        # Combine the masks on spans that are out-of-bounds, and the mask on spans that are outside
        # the top k for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        sequence_mask = util.batched_index_select(mask, top_indices, flat_top_indices)
        sequence_mask = sequence_mask.squeeze(-1).bool()
        top_mask = top_indices_mask & sequence_mask
        top_mask = top_mask.long()

        # Shape: (batch_size, max_num_items_to_keep, 1)
        top_scores = util.batched_index_select(scores, top_indices, flat_top_indices)

        return top_embeddings, top_mask, top_indices, top_scores, num_items_to_keep

    def tag_modeify_score_pruned(self, scores, raw_sentence, spans, name):
        tagged = nltk.pos_tag(raw_sentence)
        tagged_list = [tup[1] for tup in tagged]
        for i in range(0, spans.shape[1]):
            span = spans[0][i]
            span_tag_list = tagged_list[span[0]:span[1] + 1]
            if name == 'target':
                if 'NN' in span_tag_list  \
                        or 'NNS' in span_tag_list or 'NNP' in span_tag_list or 'NNPS' in span_tag_list:
                    scores[0][i][0] = scores[0][i][0] * 1.1
                if 'MD' in span_tag_list or '.' in span_tag_list or ':' in span_tag_list\
                        or ',' in span_tag_list or 'TO' in span_tag_list:
                    scores[0][i][0] = scores[0][i][0] * 0.9
            elif name == 'opinion':
                if  'JJ' in span_tag_list or 'JJR' in span_tag_list \
                        or 'JJS' in span_tag_list:
                    scores[0][i][0] = scores[0][i][0] * 1.05
                if 'CD' in span_tag_list or '.' in span_tag_list or ':' in span_tag_list\
                        or ',' in span_tag_list:
                    scores[0][i][0] = scores[0][i][0] * 0.9
        return scores

    def tag_dec_num_pruned(self, top_indices,raw_sentence,spans,name):
        new_top_indices = []
        tagged = nltk.pos_tag(raw_sentence)
        tagged_list = [tup[1] for tup in tagged]
        for i in range(0, top_indices.shape[1]):
            indice = top_indices[0][i][0].item()
            span = spans[0][indice]
            span_tag_list = tagged_list[span[0]:span[1] + 1]
            if name == 'target':
                if ',' in span_tag_list  \
                        or 'TO' in span_tag_list or ':' in span_tag_list or '.' in span_tag_list:
                    continue
                else:
                    new_top_indices.append(indice)
            elif name == 'opinion':
                if ',' in span_tag_list or 'CC' in span_tag_list or 'TO' in span_tag_list \
                        or ':' in span_tag_list or '.' in span_tag_list:
                    continue
                else:
                    new_top_indices.append(indice)
        if len(new_top_indices) == 0:
            return None,None,None
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            answer_num_items_to_keep = torch.LongTensor([len(new_top_indices),])
            answer_num_items_to_keep = answer_num_items_to_keep.to(device)
            answer_max_max_items_to_keep = max(answer_num_items_to_keep.max().item(), 1)
            answer_top_indices = torch.LongTensor(new_top_indices).reshape(1,-1,1)
            answer_top_indices = answer_top_indices.to(device)
            return answer_num_items_to_keep,answer_max_max_items_to_keep,answer_top_indices


class TwoScorePruner(torch.nn.Module):
    """
    Output has 2 columns instead of 1
    So that we have a invalid/valid score for spans separately
    This way we can add the "invalid" span score to "invalid" relation score
    And add the "valid" span score to pos/neg/neu relation score
    Internally we normalize both columns and take 1 for top-k sorting
    But output span scores should be un-normalized logits
    """
    def __init__(self, scorer: torch.nn.Module) -> None:
        super().__init__()
        self._scorer = scorer
        self.output_size = 2

     
    def forward(
        self,  # pylint: disable=arguments-differ
        embeddings: torch.FloatTensor,
        mask: torch.LongTensor,
        num_items_to_keep: Union[int, torch.LongTensor],
    ) -> Tuple[
        torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor
    ]:
        # If an int was given for number of items to keep, construct tensor by repeating the value.
        if isinstance(num_items_to_keep, int):
            batch_size = mask.size(0)
            # Put the tensor on same device as the mask.
            num_items_to_keep = num_items_to_keep * torch.ones(
                [batch_size], dtype=torch.long, device=mask.device
            )

        mask = mask.unsqueeze(-1)
        num_items = embeddings.size(1)
        output_scores = self._scorer(embeddings)
        assert output_scores.shape[-1] == self.output_size
        scores = output_scores.softmax(dim=-1)[..., [1]]  # Normalize for sorting

        # Always keep at least one item to avoid edge case with empty matrix.
        max_items_to_keep = max(num_items_to_keep.max().item(), 1)

        if scores.size(-1) != 1 or scores.dim() != 3:
            raise ValueError(
                f"The scorer passed to Pruner must produce a tensor of shape"
                f"(batch_size, num_items, 1), but found shape {scores.size()}"
            )
        # Make sure that we don't select any masked items by setting their scores to be very
        # negative.  These are logits, typically, so -1e20 should be plenty negative.
        # NOTE(`mask` needs to be a byte tensor now.)
        scores = util.replace_masked_values(scores, mask.bool(), -1e20)

        # Shape: (batch_size, max_num_items_to_keep, 1)
        _, top_indices = scores.topk(max_items_to_keep, 1)

        # Mask based on number of items to keep for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices_mask = util.get_mask_from_sequence_lengths(
            num_items_to_keep, max_items_to_keep
        )
        top_indices_mask = top_indices_mask.bool()

        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = top_indices.squeeze(-1)

        # Fill all masked indices with largest "top" index for that sentence, so that all masked
        # indices will be sorted to the end.
        # Shape: (batch_size, 1)
        fill_value, _ = top_indices.max(dim=1)
        fill_value = fill_value.unsqueeze(-1)
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = torch.where(top_indices_mask, top_indices, fill_value)

        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``embeddings`` tensor).
        top_indices, _ = torch.sort(top_indices, 1)

        # Shape: (batch_size * max_num_items_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select items for each element in the batch.
        flat_top_indices = util.flatten_and_batch_shift_indices(top_indices, num_items)

        # Shape: (batch_size, max_num_items_to_keep, embedding_size)
        top_embeddings = util.batched_index_select(
            embeddings, top_indices, flat_top_indices
        )

        # Combine the masks on spans that are out-of-bounds, and the mask on spans that are outside
        # the top k for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        sequence_mask = util.batched_index_select(mask, top_indices, flat_top_indices)
        sequence_mask = sequence_mask.squeeze(-1).bool()
        top_mask = top_indices_mask & sequence_mask
        top_mask = top_mask.long()

        # Shape: (batch_size, max_num_items_to_keep, 1)
        top_scores = util.batched_index_select(output_scores, top_indices, flat_top_indices)

        return top_embeddings, top_mask, top_indices, top_scores, num_items_to_keep


class ClassifyMaskPruner(Pruner):
    def __init__(self, scorer: torch.nn.Module, threshold=0.5, **kwargs):
        super().__init__(scorer, **kwargs)
        self._threshold = threshold

     
    def forward(
            self,  # pylint: disable=arguments-differ
            embeddings: torch.FloatTensor,
            mask: torch.LongTensor,
            num_items_to_keep: Union[int, torch.LongTensor],
            class_scores: torch.FloatTensor = None,
            gold_labels: torch.long = None,
            extra_scores: torch.FloatTensor = None,  # Scores to add to scorer output
    ) -> Tuple[
        torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor
    ]:
        mask = mask.unsqueeze(-1)
        scores = self._scorer(embeddings)
        bs, num_items, size = scores.shape
        assert size == 1
        if extra_scores is not None:
            # Assumes extra_scores is already in [0, 1] range
            scores = scores.sigmoid() + extra_scores

        # Make sure that we don't select any masked items by setting their scores to be very
        # negative.  These are logits, typically, so -1e20 should be plenty negative.
        # NOTE(`mask` needs to be a byte tensor now.)
        scores = util.replace_masked_values(scores, mask.bool(), -1e20)

        keep = torch.gt(scores.sigmoid(), self._threshold).long()
        num_items_to_keep = keep.sum(dim=1).view(bs)
        num_items_to_keep = torch.clamp(num_items_to_keep, min=1)
        # import logging
        # logging.info(dict(num_items_to_keep=num_items_to_keep))

        # Always keep at least one item to avoid edge case with empty matrix.
        max_items_to_keep = max(num_items_to_keep.max().item(), 1)

        # Shape: (batch_size, max_num_items_to_keep, 1)
        _, top_indices = scores.topk(max_items_to_keep, 1)

        # Mask based on number of items to keep for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices_mask = util.get_mask_from_sequence_lengths(
            num_items_to_keep, max_items_to_keep
        )
        top_indices_mask = top_indices_mask.bool()

        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = top_indices.squeeze(-1)

        # Fill all masked indices with largest "top" index for that sentence, so that all masked
        # indices will be sorted to the end.
        # Shape: (batch_size, 1)
        fill_value, _ = top_indices.max(dim=1)
        fill_value = fill_value.unsqueeze(-1)
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = torch.where(top_indices_mask, top_indices, fill_value)

        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``embeddings`` tensor).
        top_indices, _ = torch.sort(top_indices, 1)

        # Shape: (batch_size * max_num_items_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select items for each element in the batch.
        flat_top_indices = util.flatten_and_batch_shift_indices(top_indices, num_items)

        # Shape: (batch_size, max_num_items_to_keep, embedding_size)
        top_embeddings = util.batched_index_select(
            embeddings, top_indices, flat_top_indices
        )

        # Combine the masks on spans that are out-of-bounds, and the mask on spans that are outside
        # the top k for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        sequence_mask = util.batched_index_select(mask, top_indices, flat_top_indices)
        sequence_mask = sequence_mask.squeeze(-1).bool()
        top_mask = top_indices_mask & sequence_mask
        top_mask = top_mask.long()

        # Shape: (batch_size, max_num_items_to_keep, 1)
        top_scores = util.batched_index_select(scores, top_indices, flat_top_indices)

        return top_embeddings, top_mask, top_indices, top_scores, num_items_to_keep
