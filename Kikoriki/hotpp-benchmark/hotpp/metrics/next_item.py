import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from .tmap import compute_map


class NextItemMetric(Metric):
    """Next item (event) prediction evaluation metrics.

    Computes:
      - next-item-mean-time-step
      - next-item-mae
      - next-item-rmse
      - next-item-accuracy
      - next-item-max-f-score (+ weighted)
      - next-item-map (+ weighted)
      - next-item-hr@K
      - next-item-mrr@K
      - next-item-ndcg@K
    """

    def __init__(self, max_time_delta=None, compute_on_cpu=False, topk=(1, 5, 10, 20)):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.max_time_delta = max_time_delta
        self._device = torch.device("cpu")

        # ranking ks
        self.topk = tuple(int(k) for k in topk)
        if any(k <= 0 for k in self.topk):
            raise ValueError(f"topk must be positive, got {self.topk}")
        self._max_k = max(self.topk)

        # There is a bug with tensor states, when computed on CPU. Use lists instead.
        self.add_state("_n_correct_labels", default=[], dist_reduce_fx="cat")
        self.add_state("_n_labels", default=[], dist_reduce_fx="cat")
        self.add_state("_scores", default=[], dist_reduce_fx="cat")
        self.add_state("_labels", default=[], dist_reduce_fx="cat")
        self.add_state("_ae_sums", default=[], dist_reduce_fx="cat")
        self.add_state("_se_sums", default=[], dist_reduce_fx="cat")
        self.add_state("_delta_sums", default=[], dist_reduce_fx="cat")
        self.add_state("_n_deltas", default=[], dist_reduce_fx="cat")

        # ranking metric states
        self.add_state("_n_rank", default=[], dist_reduce_fx="cat")
        for k in self.topk:
            self.add_state(f"_hr_sum_{k}", default=[], dist_reduce_fx="cat")
            self.add_state(f"_mrr_sum_{k}", default=[], dist_reduce_fx="cat")
            self.add_state(f"_ndcg_sum_{k}", default=[], dist_reduce_fx="cat")

    def update(
        self,
        mask,
        target_timestamps,
        target_labels,
        predicted_timestamps,
        predicted_labels,
        predicted_labels_logits,
    ):
        """Update metrics with new data.

        Args:
            mask: Valid targets and predictions mask with shape (B, L).
            target_timestamps: Valid target timestamps with shape (B, L).
            target_labels: True labels with shape (B, L).
            predicted_timestamps: Predicted timestamps with shape (B, L).
            predicted_labels: Predicted labels with shape (B, L).
            predicted_labels_logits: Predicted class logits with shape (B, L, C).
        """
        device = mask.device

        # ---- classification-style metrics for next item ----
        is_correct = predicted_labels == target_labels  # (B, L)
        is_correct = is_correct.masked_select(mask)  # (V)
        self._n_correct_labels.append(torch.tensor([is_correct.sum()], device=device))
        self._n_labels.append(torch.tensor([is_correct.numel()], device=device))
        self._scores.append(predicted_labels_logits[mask])  # (V, C)
        self._labels.append(target_labels[mask])  # (V)

        # ---- time errors ----
        ae = (target_timestamps - predicted_timestamps).abs()  # (B, L)
        if self.max_time_delta is not None:
            ae = ae.clip(max=self.max_time_delta)
        ae = ae.masked_select(mask)  # (V)
        assert ae.numel() == is_correct.numel()
        self._ae_sums.append(ae.float().mean(0, keepdim=True) * ae.numel())
        self._se_sums.append(ae.square().float().mean(0, keepdim=True) * ae.numel())

        # ---- mean predicted time step ----
        deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]  # (B, L-1)
        deltas = deltas.clip(min=0)
        deltas = deltas.masked_select(torch.logical_and(mask[:, 1:], mask[:, :-1]))  # (V)
        self._delta_sums.append(deltas.float().mean(0, keepdim=True) * deltas.numel())
        self._n_deltas.append(torch.tensor([deltas.numel()], device=device))

        # ---- ranking metrics (HR/MRR/NDCG) ----
        scores_v = predicted_labels_logits[mask]  # (V, C)
        labels_v = target_labels[mask].long()  # (V,)
        if scores_v.numel() > 0:
            max_k = min(self._max_k, scores_v.shape[-1])
            topk_idx = scores_v.topk(max_k, dim=-1).indices  # (V, max_k)
            match = topk_idx.eq(labels_v.unsqueeze(-1))  # (V, max_k)

            # position of true label in topk (0-based), or max_k if absent
            pos_idx = torch.arange(max_k, device=device).unsqueeze(0).expand_as(match)
            pos = torch.where(match, pos_idx, torch.full_like(pos_idx, max_k)).min(dim=-1).values  # (V,)

            self._n_rank.append(torch.tensor([scores_v.shape[0]], device=device))

            for k in self.topk:
                kk = min(k, max_k)
                hit = (pos < kk).float()  # (V,)

                hr_sum = hit.sum().view(1)
                mrr_sum = torch.where(
                    pos < kk,
                    1.0 / (pos.float() + 1.0),
                    torch.zeros_like(pos, dtype=torch.float),
                ).sum().view(1)
                ndcg_sum = torch.where(
                    pos < kk,
                    1.0 / torch.log2(pos.float() + 2.0),  # log2(rank+1), rank = pos+1
                    torch.zeros_like(pos, dtype=torch.float),
                ).sum().view(1)

                getattr(self, f"_hr_sum_{k}").append(hr_sum)
                getattr(self, f"_mrr_sum_{k}").append(mrr_sum)
                getattr(self, f"_ndcg_sum_{k}").append(ndcg_sum)

        self._device = mask.device

    def compute(self):
        delta_sums = dim_zero_cat(self._delta_sums)
        if len(delta_sums) == 0:
            return {}

        device = delta_sums.device
        ae_sums = dim_zero_cat(self._ae_sums)
        se_sums = dim_zero_cat(self._se_sums)
        scores = dim_zero_cat(self._scores)
        n_deltas = dim_zero_cat(self._n_deltas).sum().item()
        n_labels = dim_zero_cat(self._n_labels).sum().item()
        n_correct_labels = dim_zero_cat(self._n_correct_labels).sum().item()

        # MAP / max-F from per-class logits
        nc = scores.shape[-1]
        labels = dim_zero_cat(self._labels)
        one_hot_labels = torch.nn.functional.one_hot(labels.long(), nc).bool()  # (V, C)
        micro_weights = one_hot_labels.sum(0) / one_hot_labels.sum()  # (C)
        aps, max_f_scores = compute_map(one_hot_labels, scores, device=self._device)  # (C)
        aps = aps.to(device)
        max_f_scores = max_f_scores.to(device)

        # ranking metrics aggregation
        ranking = {}
        n_rank = dim_zero_cat(self._n_rank).sum().item() if len(self._n_rank) > 0 else 0
        if n_rank > 0:
            for k in self.topk:
                hr = dim_zero_cat(getattr(self, f"_hr_sum_{k}")).sum().item() / n_rank
                mrr = dim_zero_cat(getattr(self, f"_mrr_sum_{k}")).sum().item() / n_rank
                ndcg = dim_zero_cat(getattr(self, f"_ndcg_sum_{k}")).sum().item() / n_rank
                ranking[f"next-item-hr@{k}"] = hr
                ranking[f"next-item-mrr@{k}"] = mrr
                ranking[f"next-item-ndcg@{k}"] = ndcg

        return {
            "next-item-mean-time-step": delta_sums.sum().item() / max(n_deltas, 1),
            "next-item-mae": ae_sums.sum().item() / max(n_labels, 1),
            "next-item-rmse": (se_sums.sum() / max(n_labels, 1)).sqrt().item(),
            "next-item-accuracy": n_correct_labels / max(n_labels, 1),
            "next-item-max-f-score": max_f_scores.mean().item(),
            "next-item-max-f-score-weighted": (max_f_scores * micro_weights).sum().item(),
            "next-item-map": aps.mean().item(),
            "next-item-map-weighted": (aps * micro_weights).sum().item(),
            **ranking,
        }
