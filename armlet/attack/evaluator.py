import numpy as np
import torch
import hydra

from armlet.attack.data import construct_attack_test_dataloader
from armlet.attack.metrics import (
    best_balanced_attack_success_rate,
    roc_auc_score,
    tpr_at_fpr,
)


class MIAEvaluator():

    def __init__(
        self,
        eval_every: int,
        eval_min_epoch: int,
        tpr_fpr,
        best_balanced_asr: bool,
        delta_metrics: bool,
        train_set,
        test_set,
        attack_cfg,
        **kwargs,
    ):

        self.eval_every = eval_every
        self.eval_min_epoch = eval_min_epoch
        self.fpr_target = tpr_fpr
        self.best_balanced_asr = best_balanced_asr

        self.delta_metrics = delta_metrics
        self._previous_scores = None
        self._previous_logits = None
        self._previous_model_vector = None

        self.attack_data_loader = construct_attack_test_dataloader(
            train_set,
            test_set,
        )

        self.attack = hydra.utils.instantiate(
            attack_cfg,
            _recursive_=False,
            **kwargs,
        )

        self.attack.prepare_attack()

    @torch.no_grad
    def evaluate(
        self,
        round: int,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> dict:

        if (round != 1) and (round % self.eval_every != 0):
            return {}

        if (model is None) or (self.attack_data_loader is None):
            return {}

        if round < self.eval_min_epoch:
            return {}

        y_score, y_true, logits = self.attack.infer_membership(model, device, self.attack_data_loader)

        attack_metrics_values = self._compute_attack_metrics_values(y_true, y_score)
        result = {m: np.round(v, 5).item() for m, v in attack_metrics_values.items()}

        if self.delta_metrics:
            model_vector = _flatten_model_params(model)
            delta_metrics_values = self._compute_delta_metrics_values(y_score, logits, model_vector)
            result.update({m: np.round(v, 8).item() for m, v in delta_metrics_values.items()})

        # add leakeage metric here based on self.attack_data_loader and model

        result = {f"attack_{m}": v for m, v in result.items()}
        return result

    def _compute_attack_metrics_values(self, y_true, y_score):

        attack_metrics_values = {}

        if self.best_balanced_asr:
            asr = best_balanced_attack_success_rate(y_true, y_score)
        else:
            asr = float(((y_score >= 0.5).astype(int) == y_true).mean())
        attack_metrics_values["asr"] = asr

        attack_metrics_values["auc"] = roc_auc_score(y_true, y_score)
        attack_metrics_values["tpr_at_fpr"] = tpr_at_fpr(y_true, y_score, self.fpr_target)
        attack_metrics_values["score_mean"] = float(np.mean(y_score))
        attack_metrics_values["score_std"] = float(np.std(y_score))

        return attack_metrics_values

    def _compute_delta_metrics_values(self, y_score, logits, model_vector):

        delta_metrics_values = {}

        if self._previous_scores is not None and self._previous_scores.shape == y_score.shape:
            score_delta = float(np.linalg.norm(
                y_score - self._previous_scores) / max(1, y_score.size))
        else:
            score_delta = 0.0
        delta_metrics_values["score_delta"] = score_delta
        self._previous_scores = y_score.copy()

        if self._previous_logits is not None and self._previous_logits.shape == logits.shape:
            logit_delta = float(np.linalg.norm(
                logits - self._previous_logits) / max(1, logits.size))
        else:
            logit_delta = 0.0
        delta_metrics_values["logit_delta"] = logit_delta
        self._previous_logits = logits.copy()

        if self._previous_model_vector is not None and self._previous_model_vector.shape == model_vector.shape:
            model_delta = float(torch.linalg.vector_norm(
                model_vector - self._previous_model_vector).item() / max(1, model_vector.numel()))
        else:
            model_delta = 0.0
        delta_metrics_values["model_delta"] = model_delta
        self._previous_model_vector = model_vector

        return delta_metrics_values


def _flatten_model_params(model: torch.nn.Module) -> torch.Tensor:
    params = [p.detach().cpu().reshape(-1) for p in model.parameters()]
    if not params:
        return torch.empty(0)
    return torch.cat(params)
