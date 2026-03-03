from typing import Dict
import torch
from torch import Tensor
from torchmetrics import Metric


class BinaryFairnessMetrics(Metric):
    def __init__(self, sensitive_attribute: str, **kwargs):
        super().__init__(**kwargs)

        self.add_state("tp_group_A", default = torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("fp_group_A", default = torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("tn_group_A", default = torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("fn_group_A", default = torch.tensor(0), dist_reduce_fx = "sum")

        self.add_state("tp_group_B", default = torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("fp_group_B", default = torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("tn_group_B", default = torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("fn_group_B", default = torch.tensor(0), dist_reduce_fx = "sum")

        self.metrics_names = [
            '{}_spd'.format(sensitive_attribute),
            '{}_disparate_impact'.format(sensitive_attribute),
            '{}_discr_index'.format(sensitive_attribute),
            '{}_eod'.format(sensitive_attribute),
            '{}_avg_odds'.format(sensitive_attribute),
        ]

    def update(self, preds: Tensor, target: Tensor, sensitive_data: Tensor):

        assert not preds.isnan().any()
        if preds.is_floating_point():
            if not torch.all((preds >= 0) * (preds <= 1)):
                preds = preds.sigmoid()
            preds = preds > 0.5

        preds = torch.flatten(preds)
        target = torch.flatten(target)

        self.tp_group_A += torch.sum((preds == 1) * (preds == target) * (sensitive_data == 1))
        self.fp_group_A += torch.sum((preds == 1) * (preds != target) * (sensitive_data == 1))
        self.tn_group_A += torch.sum((preds == 0) * (preds == target) * (sensitive_data == 1))
        self.fn_group_A += torch.sum((preds == 0) * (preds != target) * (sensitive_data == 1))

        self.tp_group_B += torch.sum((preds == 1) * (preds == target) * (sensitive_data == 0))
        self.fp_group_B += torch.sum((preds == 1) * (preds != target) * (sensitive_data == 0))
        self.tn_group_B += torch.sum((preds == 0) * (preds == target) * (sensitive_data == 0))
        self.fn_group_B += torch.sum((preds == 0) * (preds != target) * (sensitive_data == 0))

    def compute(self) -> Dict[str, Tensor]:

        positive_rate_group_A = div_default_zero(
            torch.add(self.tp_group_A, self.fp_group_A),
            torch.add(self.tp_group_A, self.fp_group_A) + torch.add(self.tn_group_A, self.fn_group_A)
        )
        tpr_group_A = div_default_zero(self.tp_group_A, torch.add(self.tp_group_A, self.fn_group_A))
        fpr_group_A = div_default_zero(self.fp_group_A, torch.add(self.fp_group_A, self.tn_group_A))
        precision_group_A = div_default_zero(self.tp_group_A, torch.add(self.tp_group_A, self.fp_group_A))
        f1_group_A = 2*div_default_zero(precision_group_A * tpr_group_A, torch.add(precision_group_A, tpr_group_A))
    
        positive_rate_group_B = div_default_zero(
            torch.add(self.tp_group_B, self.fp_group_B),
            torch.add(self.tp_group_B, self.fp_group_B) + torch.add(self.tn_group_B, self.fn_group_B)
        )
        tpr_group_B = div_default_zero(self.tp_group_B, torch.add(self.tp_group_B, self.fn_group_B))
        fpr_group_B = div_default_zero(self.fp_group_B, torch.add(self.fp_group_B, self.tn_group_B))
        precision_group_B = div_default_zero(self.tp_group_B, torch.add(self.tp_group_B, self.fp_group_B))
        f1_group_B = 2*div_default_zero(precision_group_B * tpr_group_B, torch.add(precision_group_B, tpr_group_B))

        spd =  positive_rate_group_A - positive_rate_group_B

        if positive_rate_group_B == 0:
            disparate_impact = torch.tensor(0) # not 0, but np.inf?
        else:
            disparate_impact = positive_rate_group_A/positive_rate_group_B

        discr_index = f1_group_A - f1_group_B

        eod = tpr_group_A - tpr_group_B

        avg_odds = ((fpr_group_A - fpr_group_B) + (tpr_group_A - tpr_group_B))/2

        return {
            self.metrics_names[0]: spd,
            self.metrics_names[1]: disparate_impact,
            self.metrics_names[2]: discr_index,
            self.metrics_names[3]: eod,
            self.metrics_names[4]: avg_odds,
        }

def div_default_zero(positive_preds: Tensor, total_preds: Tensor) -> Tensor:
    if total_preds > 0:
        return positive_preds/total_preds
    else:
        return torch.tensor(0)
