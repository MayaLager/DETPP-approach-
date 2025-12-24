import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class NextItemRankingMetric(Metric):
    """
    Ranking-only next-item metrics:
      - next-item-hr@K
      - next-item-mrr@K
      - next-item-ndcg@K

    Expects update(...) with:
      mask: (B, L) bool
      target_labels: (B, L) long
      predicted_labels_logits: (B, L, C) float
    Other args are accepted for API compatibility and ignored.
    """

    def __init__(self, compute_on_cpu: bool = False, topk=(1, 5, 10, 20)):
        super().__init__(compute_on_cpu=compute_on_cpu)

        self.topk = tuple(int(k) for k in topk)
        if len(self.topk) == 0 or any(k <= 0 for k in self.topk):
            raise ValueError(f"topk must be positive, got {self.topk}")
        self._max_k = max(self.topk)

        # Use lists (same workaround style as in your NextItemMetric)
        self.add_state("_n_rank", default=[], dist_reduce_fx="cat")
        for k in self.topk:
            self.add_state(f"_hr_sum_{k}", default=[], dist_reduce_fx="cat")
            self.add_state(f"_mrr_sum_{k}", default=[], dist_reduce_fx="cat")
            self.add_state(f"_ndcg_sum_{k}", default=[], dist_reduce_fx="cat")

        self._device = torch.device("cpu")

    def update(
        self,
        mask,
        target_timestamps=None,          # ignored
        target_labels=None,
        predicted_timestamps=None,       # ignored
        predicted_labels=None,           # ignored
        predicted_labels_logits=None,
    ):
        device = mask.device
        self._device = device

        if target_labels is None or predicted_labels_logits is None:
            return

        # Flatten valid positions
        scores_v = predicted_labels_logits[mask]  # (V, C)
        labels_v = target_labels[mask].long()     # (V,)

        if scores_v.numel() == 0:
            return

        max_k = min(self._max_k, scores_v.shape[-1])
        topk_idx = scores_v.topk(max_k, dim=-1).indices  # (V, max_k)
        match = topk_idx.eq(labels_v.unsqueeze(-1))      # (V, max_k)

        pos_idx = torch.arange(max_k, device=device).unsqueeze(0).expand_as(match)
        pos = torch.where(match, pos_idx, torch.full_like(pos_idx, max_k)).min(dim=-1).values  # (V,)

        self._n_rank.append(torch.tensor([scores_v.shape[0]], device=device))

        for k in self.topk:
            kk = min(k, max_k)
            hit = (pos < kk).float()

            hr_sum = hit.sum().view(1)
            mrr_sum = torch.where(
                pos < kk,
                1.0 / (pos.float() + 1.0),
                torch.zeros_like(pos, dtype=torch.float),
            ).sum().view(1)
            ndcg_sum = torch.where(
                pos < kk,
                1.0 / torch.log2(pos.float() + 2.0),
                torch.zeros_like(pos, dtype=torch.float),
            ).sum().view(1)

            getattr(self, f"_hr_sum_{k}").append(hr_sum)
            getattr(self, f"_mrr_sum_{k}").append(mrr_sum)
            getattr(self, f"_ndcg_sum_{k}").append(ndcg_sum)

    def compute(self):
        n_rank = dim_zero_cat(self._n_rank).sum().item() if len(self._n_rank) > 0 else 0
        if n_rank <= 0:
            return {}

        out = {}
        for k in self.topk:
            hr = dim_zero_cat(getattr(self, f"_hr_sum_{k}")).sum().item() / n_rank
            mrr = dim_zero_cat(getattr(self, f"_mrr_sum_{k}")).sum().item() / n_rank
            ndcg = dim_zero_cat(getattr(self, f"_ndcg_sum_{k}")).sum().item() / n_rank
            out[f"next-item-hr@{k}"] = hr
            out[f"next-item-mrr@{k}"] = mrr
            out[f"next-item-ndcg@{k}"] = ndcg

        return out
