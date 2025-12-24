import torch

from .next_item import NextItemMetric
from .horizon import HorizonMetric


class HorizonWithNextItemMetric(HorizonMetric):
    """
    HorizonMetric (DeTPP: T-mAP/OTD/...) + NextItemMetric (твои next-item метрики по global target).
    """

    def __init__(
        self,
        next_item_topk=(1, 5, 10, 20),
        next_item_max_time_delta=None,
        next_item_compute_on_cpu=False,
        **horizon_kwargs,
    ):
        # IMPORTANT:
        # During super().__init__(), torchmetrics/HorizonMetric may call reset().
        # Our reset() references _next_item, so we must define it BEFORE super().__init__().
        self._next_item = None

        super().__init__(**horizon_kwargs)

        # Now it's safe to create the inner metric.
        self._next_item = NextItemMetric(
            max_time_delta=next_item_max_time_delta,
            compute_on_cpu=next_item_compute_on_cpu,
            topk=next_item_topk,
        )

    @torch.no_grad()
    def update_global_target(self, lengths, targets, predicted_timestamps, predicted_labels, predicted_logits):
        """
        Считает NextItemMetric по targets.payload['target_labels','target_timestamps'].

        Берём предсказание модели на последней позиции контекста (индекс lengths-1).
        """
        if self._next_item is None:
            return

        if ("target_labels" not in targets.payload) or ("target_timestamps" not in targets.payload):
            raise RuntimeError(
                "Need targets.payload['target_labels'] and ['target_timestamps'] for NextItemMetric. "
                "Add: +data_module.{val,test}_params.global_target_fields='[target_labels,target_timestamps]'."
            )

        tgt_y = targets.payload["target_labels"]
        tgt_t = targets.payload["target_timestamps"]

        # (B,) -> (B,1)
        if tgt_y.ndim == 1:
            tgt_y = tgt_y.unsqueeze(1)
        if tgt_t.ndim == 1:
            tgt_t = tgt_t.unsqueeze(1)

        mask = (tgt_y != -1) & torch.isfinite(tgt_t)
        mask = mask.bool()

        bsz = predicted_labels.shape[0]
        device = predicted_labels.device

        last_idx = (lengths.to(torch.long) - 1).clamp(min=0)  # (B,) long
        ar = torch.arange(bsz, device=device, dtype=torch.long)

        pred_t_last = predicted_timestamps[ar, last_idx].unsqueeze(1)   # (B,1)
        pred_y_last = predicted_labels[ar, last_idx].unsqueeze(1)       # (B,1)
        pred_logits_last = predicted_logits[ar, last_idx].unsqueeze(1)  # (B,1,C)

        self._next_item.update(
            mask=mask.to(device),
            target_timestamps=tgt_t.to(device).float(),
            target_labels=tgt_y.to(device).long(),
            predicted_timestamps=pred_t_last.float(),
            predicted_labels=pred_y_last.long(),
            predicted_labels_logits=pred_logits_last.float(),
        )

    def compute(self):
        out = super().compute()
        if self._next_item is not None:
            out.update(self._next_item.compute())
        return out

    def reset(self):
        super().reset()
        if self._next_item is not None:
            self._next_item.reset()
