import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
improt mindspore.numpy as np

class ConLoss(nn.Cell):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.matmul = ops.MatMul()
        self.exp = ops.Exp()
        self.log = ops.Log()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.sub = ops.Sub()
        self.mul = ops.Mul()
        self.div = ops.Div()
        self.eye = ops.Eye()
        self.max = ops.ReduceMax()
        self.eq = ops.Equal()
        self.concat = ops.Concat(axis=1)

    def construct(self, features, labels=None, mask=None):
        device = mindspore.context.get_context('device_target')
        features = ops.Normalize()(features, p=2, axis=1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = self.eye(batch_size, dtype=mindspore.float32).to(device)
        elif labels is not None:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = self.eq(labels, labels.transpose()).float().to(device)
        else:
            mask = mask.float().to(device)

        features_T = features.transpose()
        anchor_dot_contrast = self.div(self.matmul(features, features_T), self.temperature)
        logits_max, _ = self.max(anchor_dot_contrast, axis=1, keep_dims=True)
        logits = self.sub(anchor_dot_contrast, logits_max.detach())
        exp_logits = self.exp(logits)

        logits_mask = self.sub(1.0, self.eye(batch_size))
        positives_mask = self.mul(mask, logits_mask)
        negatives_mask = self.sub(1.0, mask)

        num_positives_per_row = self.sum(positives_mask, axis=1)
        denominator = self.sum(self.mul(exp_logits, negatives_mask), axis=1, keep_dims=True) + \
                      self.sum(self.mul(exp_logits, positives_mask), axis=1, keep_dims=True)

        log_probs = self.sub(logits, self.log(denominator))
        if self.mul(log_probs, log_probs).any():
            raise ValueError("Log_prob has nan!")

        log_probs = self.sum(self.mul(log_probs, positives_mask), axis=1)[num_positives_per_row > 0] / \
                    num_positives_per_row[num_positives_per_row > 0]

        loss = self.mul(self.mean(log_probs), -1.0)
        if self.scale_by_temperature:
            loss = self.mul(loss, self.temperature)

        return loss


class KL_div_loss(nn.Cell):
    def __init__(self):
        super(KL_div_loss, self).__init__()

    def construct(self, input, target):
        input = ops.Log()(input)
        target = target * ops.Log()(target)
        loss = ops.ReduceSum()(target - input, (0, 1))
        return loss

class ContrastiveLoss(nn.Cell):
    def __init__(self, t=0.07):
        super(ContrastiveLoss, self).__init__()
        self.t = t

    def construct(self, feats):
        feats = ops.Normalize(axis=2)(feats)  # B x K x C
        scores = ops.Einsum('aid, bjd -> abij', feats, feats)
        scores = ops.Rearrange('a b i j -> (a i) (b j)')(scores)

        # positive logits: Nx1
        pos_idx = ops.Fill()(dtype=mindspore.int32, dims=(feats.shape[1], feats.shape[1]), value=0)
        pos_idx = ops.Eye()(pos_idx, diagonal=0)
        pos_idx = ops.Repeat()(pos_idx, 'i j -> (a i) (b j)', a=feats.shape[0], b=feats.shape[0])
        pos_idx = ops.FillDiagonal()(pos_idx, 0)
        l_pos = ops.Gather()(scores, 1, ops.NonZero()(pos_idx)[:, 1].view(scores.shape[0], -1))
        rand_idx = ops.RandomInt()(1, l_pos.shape[1], (l_pos.shape[0], 1))
        l_pos = ops.Gather()(l_pos, 1, rand_idx)

        # negative logits: NxK
        neg_idx = ops.Fill()(dtype=mindspore.int32, dims=(feats.shape[1], feats.shape[1]), value=1)
        neg_idx = ops.Sub()(neg_idx, ops.Eye()(neg_idx, diagonal=0))
        neg_idx = ops.Repeat()(neg_idx, 'i j -> (a i) (b j)', a=feats.shape[0], b=feats.shape[0])
        l_neg = ops.Gather()(scores, 1, ops.NonZero()(neg_idx)[:, 1].view(scores.shape[0], -1))
        # logits: Nx(1+K)
        logits = ops.Concat(1)([l_pos, l_neg])

        # apply temperature
        logits /= self.t

        # labels: positive key indicators
        labels = ops.Zeros()(logits.shape[0], dtype=mindspore.int32)
        return nn.CrossEntropyLoss()(logits, labels)


class ConsistencyLoss(nn.Cell):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def construct(self, masks, image, fg):
        # masks: N x R x H x W
        # image: N x 3 x H x W
        weighted_regions = ops.Unsqueeze(2)(masks) * ops.Unsqueeze(1)(image)  # N x R x 3 x H x W
        mask_sum = ops.Sum()(masks, axis=3).sum(2, keepdims=True)  # N x R x 1
        means = ops.Sum()(weighted_regions, axis=4).sum(3) / (mask_sum + 1e-5)  # N x R x 3
        diff_sq = (ops.Unsqueeze(1)(image) - ops.Unsqueeze(3)(ops.Unsqueeze(4)(means)))**2  # N x R x 3 x H x W
        loss = (diff_sq * ops.Unsqueeze(2)(masks) * ops.Unsqueeze(2)(fg)).sum(4).sum(3)  # N x R x 3
        loss /= (ops.Sum()(ops.Unsqueeze(2)(fg), axis=4).sum(3) + 1e-5)  # N x R x 3
        return ops.Sum()(ops.Sum()(loss, axis=2), axis=1).mean()

