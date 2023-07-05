from tqdm import tqdm
import torch
import torch.distributed as dist
from torchmetrics import Metric
from einops import repeat
from typing import Any, List
from beartype import beartype
from typing import Callable, Dict, List, Optional, Tuple, Union
from utils.math import calc_iou
from kn_util.data import to_numpy
from loguru import logger

import matplotlib.pyplot as plt
import seaborn as sns
from kn_util.basic import global_get
import kn_util.distributed as kn_dict


class ScalarMetric(Metric):
    full_state_update = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        self.sum += x
        self.num_sample += 1

    def compute(self, mode="mean"):
        if mode == "mean":
            return self.sum / self.num_sample
        elif mode == "sum":
            return self.sum
        else:
            raise ValueError(f"Unknown mode {mode}")


class RankMIoUAboveN(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, m, n) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.add_state("hit", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, timestamps_pred, timestamps_gt):
        """
        Args:
            timestamps_pred: [B, Nc, 2]
            timestamps_gt: [B, 2]
        """
        if timestamps_pred.ndim == 2:
            timestamps_pred = timestamps_pred.unsqueeze(0)  # add a mock batch dim
            timestamps_gt = timestamps_gt.unsqueeze(0)

        timestamps_pred = timestamps_pred[:, :self.m]
        B, Nc, _2 = timestamps_pred.shape

        expand_gt = repeat(timestamps_gt, "b i -> (b nc) i", nc=Nc)
        ious = calc_iou(timestamps_pred.reshape(-1, 2), expand_gt.reshape(-1, 2)).reshape(B, Nc)
        self.hit += (ious >= self.n).max(dim=1).values.sum()
        self.num_sample += B

    def compute(self):
        return self.hit / self.num_sample * 100


class Evaluater:

    @beartype
    def __init__(self, *, metrics: Dict[str, Metric] = dict(), namespace: str = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = {k: v.to(self.device) for k, v in metrics.items()}
        self.namespace = namespace

    def update_all(self, outputs):
        for metric in self.metrics.values():
            metric.update(**outputs)

    def compute_all(self):
        ret_dict = dict()
        for nm, metric in self.metrics.items():
            if self.namespace is not None:
                nm = self.namespace + "/" + nm
            ret_dict[nm] = metric.compute().item()

        return ret_dict

    def reset_all(self):
        for nm, metric in self.metrics.items():
            metric.reset()


class EvaluaterCombined(Evaluater):

    def __init__(self, evaluaters=[]):
        self.evaluaters = evaluaters

    def update_all(self, outputs):
        for evaluater in self.evaluaters:
            evaluater.update_all(outputs)

    def computer_all(self):
        ret_dict = dict()
        for evaluater in self.evaluaters:
            ret = evaluater.compute_all()
            ret_dict.update(ret)

        return ret_dict


class EvaluaterScaler(Evaluater):

    def __init__(self, *, scalers=["loss"], namespace="train"):
        metrics = {k: ScalarMetric() for k in scalers}
        super().__init__(metrics=metrics, namespace=namespace)

    def update_all(self, outputs):
        for nm, metric in self.metrics.items():
            metric.update(outputs[nm])


class EvaluaterNLVLEval(Evaluater):

    def __init__(self, *, ms, ns, namespace="eval", include_loss=True) -> None:
        metrics = {f"IoU={n:.1f}@R{m}": RankMIoUAboveN(m=m, n=n) for m in ms for n in ns}
        if include_loss:
            metrics.update(dict(loss=ScalarMetric()))
        super().__init__(metrics=metrics, namespace=namespace)

    def update_all(self, outputs):
        for nm, metric in self.metrics.items():
            if nm != "loss":
                metric.update(timestamps_pred=outputs["timestamps_pred"], timestamps_gt=outputs["gt"])
            else:
                metric.update(outputs["loss"])


@torch.no_grad()
def evaluate(model, eval_loader, evaluator):
    accelerator = global_get("accelerator")
    predictions = []
    model.eval()
    for batch in tqdm(eval_loader, desc="Evaluating", disable=not kn_dict.is_master_process()):
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(**batch)
        evaluator.update_all(dict(**outputs, **batch))
        accelerator.wait_for_everyone()

    metrics = evaluator.compute_all()

    return metrics

@torch.no_grad()
def evaluate_native(model, eval_loader, evaluator):
    predictions = []
    model.eval()
    for batch in tqdm(eval_loader, desc="Evaluating", disable=not kn_dict.is_master_process()):
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(**batch)
        evaluator.update_all(dict(**outputs, **batch))
        dist.barrier()

    metrics = evaluator.compute_all()

    return metrics