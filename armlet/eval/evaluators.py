from typing import Iterable, Optional, Union
import numpy as np
import torch
from torchmetrics import Metric, Accuracy, F1Score, Precision, Recall

from fluke.evaluation import ClassificationEval
from fluke.data import FastDataLoader

from armlet.eval.metrics import BinaryFairnessMetrics


class BinaryClassificationFairnessEval(ClassificationEval):

    def __init__(
        self,
        eval_every: int,
        n_classes: int,
        sensitive_attributes: list[str],
    ):

        super().__init__(eval_every=eval_every, n_classes=n_classes)

        assert n_classes == 2

        self.metrics = {
            "accuracy": Accuracy(
                task="binary",
                num_classes=self.n_classes,
            ),
            "precision": Precision(
                task="binary",
                num_classes=self.n_classes,
            ),
            "recall": Recall(
                task="binary",
                num_classes=self.n_classes,
            ),
            "f1": F1Score(
                task="binary",
                num_classes=self.n_classes,
            ),
        }

        self.fairness_metrics = [BinaryFairnessMetrics(sensitive_attribute) for sensitive_attribute in sensitive_attributes]

    @torch.no_grad
    def evaluate(
        self,
        round: int,
        model: torch.nn.Module,
        eval_data_loader: Union[FastDataLoader, Iterable[FastDataLoader]],
        loss_fn: Optional[torch.nn.Module] = None,
        additional_metrics: Optional[dict[str, Metric]] = None,
        device: torch.device = torch.device("cpu"), 
    ) -> dict:

        from fluke.utils import clear_cuda_cache  # NOQA

        if (round != 1) and (round % self.eval_every != 0):
            return {}

        if (model is None) or (eval_data_loader is None):
            return {}

        model_device = torch.device("cpu")
        if next(model.parameters(), None) is not None:
            model_device = next(model.parameters()).device
        model.eval()
        model.to(device)

        metrics_values = {k: [] for k in self.metrics.keys()}

        fairness_metrics_values = {}
        for fairness_metric in self.fairness_metrics:
            for metric_name in fairness_metric.metrics_names:
                fairness_metrics_values[metric_name] = []

        if additional_metrics is None:
            additional_metrics = {}
        add_metric_values = {k: [] for k in additional_metrics.keys()}

        losses = []
        cnt = 0

        if not isinstance(eval_data_loader, list):
            eval_data_loader = [eval_data_loader]

        for data_loader in eval_data_loader:

            for metric in self.metrics.values():
                metric.reset()

            for metric in additional_metrics.values():
                metric.reset()

            for fairness_metric in self.fairness_metrics:
                fairness_metric.reset()

            loss = 0

            for data in data_loader:
                
                if len(data) == 2:
                    X, y = data
                    X, y = X.to(device), y.to(device)
                    sa_tensor = X[:, -len(self.fairness_metrics):]
                else:
                    X, y, sa_tensor = data
                    X, y, sa_tensor = X.to(device), y.to(device), sa_tensor.to(device)

                with torch.no_grad():
                    y_hat = model(X)
                    if loss_fn is not None:
                        loss += loss_fn(y_hat, y).item()

                for metric in self.metrics.values():
                    metric.update(y_hat.cpu(), y.cpu())

                if additional_metrics:
                    for metric in additional_metrics.values():
                        metric.update(y_hat.cpu(), y.cpu())

                for id_sens_attr, fairness_metric in enumerate(self.fairness_metrics):
                    sensitive_data = sa_tensor[:, id_sens_attr]
                    fairness_metric.update(y_hat.cpu(), y.cpu(), sensitive_data.cpu())

            cnt += len(data_loader)

            for k, v in self.metrics.items():
                metrics_values[k].append(v.compute().item())

            if additional_metrics:
                for k, v in additional_metrics.items():
                    add_metric_values[k].append(v.compute().item())

            for fairness_metric in self.fairness_metrics:
                computed_fairness_metric = fairness_metric.compute()
                for metric_name, v in computed_fairness_metric.items():
                    fairness_metrics_values[metric_name].append(v)

            losses.append(loss / cnt)

        model.to(model_device)
        clear_cuda_cache()

        result = {m: np.round(sum(v) / len(v), 5).item() for m, v in metrics_values.items()}
        result.update({m: np.round(sum(v) / len(v), 5).item() for m, v in add_metric_values.items()})
        result.update({m: np.round(sum(v) / len(v), 5).item() for m, v in fairness_metrics_values.items()})

        if loss_fn is not None:
            result["loss"] = np.round(sum(losses) / len(losses), 5).item()

        return result
