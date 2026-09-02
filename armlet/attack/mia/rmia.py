import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

from fluke.utils import clear_cuda_cache

from armlet.attack.mia import MembershipInferenceAttack


class RMIA(MembershipInferenceAttack):

    def __init__(
        self,
        shadow_data,
        model_cfg,
        optimizer_cfg,
        loss_fn,
        device,
        shadow_train,
        signal_conversion,
        score_aggregation,
    ):

        super().__init__(
            shadow_data,
            model_cfg,
            optimizer_cfg,
            loss_fn,
            device,
            shadow_train,
        )

        self.signal_conversion_cfg = signal_conversion
        self.score_aggregation_cfg = score_aggregation

    def prepare_attack(self, attack_data_loader=None):

        if attack_data_loader is not None:
            pool_X, pool_y, membership = _attack_loader_to_tensors(attack_data_loader)
        else:
            pool_X, pool_y, membership = _loader_to_tensors(self.shadow_train_set, self.shadow_test_set)
        dataset = TensorDataset(pool_X, pool_y)

        rng = np.random.RandomState(self.shadow_train_cfg.seed)

        self.reference_models = []
        self.reference_keep_matrix = _generate_reference_keep_matrix(
            dataset_size=len(dataset),
            num_ref_models=int(self.shadow_train_cfg.num_ref_models),
            seed=self.shadow_train_cfg.seed,
        )

        self.pool_X = pool_X.detach().cpu()
        self.pool_y = pool_y.detach().cpu()
        self.pool_membership = membership.astype(bool)

        for ref_idx in range(int(self.shadow_train_cfg.num_ref_models)):
            train_idx = np.where(self.reference_keep_matrix[ref_idx])[0]
            val_size = max(1, int(len(train_idx) * self.shadow_train_cfg.ref_val_ratio))
            val_idx = rng.choice(train_idx, size=val_size, replace=False)
            batch_size = self.shadow_train_set.batch_size
            train_loader = _tensor_subset_loader(dataset, train_idx, batch_size, True)
            val_loader = _tensor_subset_loader(dataset, val_idx, batch_size, False)
            
            model_label = f"RMIA REFERENCE MODEL {ref_idx + 1}/{int(self.shadow_train_cfg.num_ref_models)}"
            reference_model = self._train_shadow_model(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                n_epochs=self.shadow_train_cfg.n_epochs,
                print_every=self.shadow_train_cfg.print_every,
                model_label=model_label,
            )
            self.reference_models.append(reference_model)

    def infer_membership(self, target_model, device, attack_data_loader):

        model_device = torch.device("cpu")
        if next(target_model.parameters(), None) is not None:
            model_device = next(target_model.parameters()).device
        target_model.eval()
        target_model.to(device)

        y_score, y_true, logits = self._score_model_on_pool(target_model, device)

        target_model.to(model_device)
        clear_cuda_cache()
        return y_score, y_true, logits

    def _score_model_on_pool(self, target_model, device):
        with torch.no_grad():
            target_logits = target_model(self.pool_X.to(device))
        target_signal = _rmia_signal(
            target_logits,
            self.pool_y.to(device),
            self.signal_conversion_cfg,
        ).cpu()

        reference_signals = []
        for model in self.reference_models:
            model.to(device)
            model.eval()
            with torch.no_grad():
                ref_logits = model(self.pool_X.to(device))
            reference_signals.append(_rmia_signal(
                ref_logits,
                self.pool_y.to(device),
                self.signal_conversion_cfg,
            ).cpu())
            model.cpu()

        scores = _aggregate_rmia_scores_for_pool(
            target_signal=target_signal,
            reference_signals=torch.stack(reference_signals, dim=0),
            reference_keep_matrix=self.reference_keep_matrix,
            membership=self.pool_membership,
            score_aggr_cfg=self.score_aggregation_cfg,
        )
        return (
            scores.detach().cpu().numpy(),
            self.pool_membership.astype(int),
            target_logits.detach().cpu().reshape(target_logits.shape[0], -1).numpy(),
        )

def _loader_to_tensors(*loaders):
    xs, ys = [], []
    for loader in loaders:
        for batch in loader:
            X, y = batch[:2]
            xs.append(X.detach().cpu())
            ys.append(y.detach().cpu())
    pool_X = torch.cat(xs, dim=0)
    pool_y = torch.cat(ys, dim=0)
    membership = np.concatenate([
        np.ones(pool_X.shape[0] // 2, dtype=bool),
        np.zeros(pool_X.shape[0] - pool_X.shape[0] // 2, dtype=bool)
    ])
    return pool_X, pool_y, membership

def _attack_loader_to_tensors(loader):
    xs, ys, memberships = [], [], []
    for batch in loader:
        X, y, attack_y = batch[:3]
        xs.append(X.detach().cpu())
        ys.append(y.detach().cpu())
        memberships.append(attack_y.detach().cpu().reshape(-1))
    return (
        torch.cat(xs, dim=0),
        torch.cat(ys, dim=0),
        torch.cat(memberships, dim=0).numpy().astype(bool),
    )

def _generate_reference_keep_matrix(dataset_size: int, num_ref_models: int, seed: int):
    rng = np.random.RandomState(seed)
    keep = np.zeros((num_ref_models, dataset_size), dtype=bool)
    n_in = max(1, num_ref_models // 2)
    for sample_idx in range(dataset_size):
        chosen = rng.choice(num_ref_models, size=n_in, replace=False)
        keep[chosen, sample_idx] = True
    return keep

def _tensor_subset_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(
        torch.utils.data.Subset(dataset, np.asarray(indices).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
    )

def _rmia_signal(logits, y, signal_conversion_cfg):
    labels = y.squeeze(-1).long().to(logits.device)
    metric = _signal_name_to_metric(signal_conversion_cfg.signal)
    extra = {
        "taylor_m": signal_conversion_cfg.taylor_m,
        "taylor_n": signal_conversion_cfg.taylor_n,
    }
    return _convert_rmia_signals(
        logits,
        labels,
        metric=metric,
        temp=signal_conversion_cfg.temperature,
        extra=extra
    ).clamp_min(1e-12).detach()

def _signal_name_to_metric(signal_name: str):
    if signal_name == "softmax_relative":
        return "softmax"
    if signal_name == "taylor_softmax_relative":
        return "taylor"
    if signal_name == "sm_softmax_relative":
        return "soft-margin"
    if signal_name == "sm_taylor_softmax_relative":
        return "taylor-soft-margin"
    raise NotImplementedError(f"Unsupported RMIA signal '{signal_name}'")

def _convert_rmia_signals(logits, true_labels, metric: str, temp: float, extra):
    if metric == "softmax":
        logit_signals = logits / temp
        logit_signals = logit_signals - torch.max(logit_signals, dim=1).values.reshape(-1, 1)
        exp_logit_signals = torch.exp(logit_signals)
        true_exp_logit = exp_logit_signals.gather(1, true_labels.reshape(-1, 1))
        output_signals = true_exp_logit / exp_logit_signals.sum(dim=1).reshape(-1, 1)
    elif metric == "taylor":
        taylor_signals = _get_taylor(logits, int(extra["taylor_n"]))
        true_taylor_logit = taylor_signals.gather(1, true_labels.reshape(-1, 1))
        output_signals = true_taylor_logit / taylor_signals.sum(dim=1).reshape(-1, 1)
    elif metric == "soft-margin":
        m = float(extra["taylor_m"])
        logit_signals = logits / temp
        exp_logit_signals = torch.exp(logit_signals)
        true_logits = logit_signals.gather(1, true_labels.reshape(-1, 1))
        exp_true_logit = exp_logit_signals.gather(1, true_labels.reshape(-1, 1))
        exp_logit_sum = exp_logit_signals.sum(dim=1).reshape(-1, 1) - exp_true_logit
        soft_true_logit = torch.exp(true_logits - m)
        output_signals = soft_true_logit / (exp_logit_sum + soft_true_logit)
    elif metric == "taylor-soft-margin":
        m = float(extra["taylor_m"])
        n = int(extra["taylor_n"])
        logit_signals = logits / temp
        taylor_logits = _get_taylor(logit_signals, n)
        true_logit = logit_signals.gather(1, true_labels.reshape(-1, 1))
        taylor_true_logit = taylor_logits.gather(1, true_labels.reshape(-1, 1))
        taylor_logit_sum = taylor_logits.sum(dim=1).reshape(-1, 1) - taylor_true_logit
        soft_taylor_true_logit = _get_taylor(true_logit - m, n)
        output_signals = soft_taylor_true_logit / (taylor_logit_sum + soft_taylor_true_logit)
    else:
        raise NotImplementedError(metric)
    return torch.flatten(output_signals)

def _get_taylor(logits, n: int):
    power = logits.clone()
    taylor = torch.ones_like(logits) + power
    factorial = 1
    for i in range(2, n + 1):
        factorial *= i
        power = power * logits
        taylor = taylor + power / factorial
    return taylor

def _aggregate_rmia_scores_for_pool(
    target_signal,
    reference_signals,
    reference_keep_matrix,
    membership,
    score_aggr_cfg,
):

    ref_keep = torch.from_numpy(reference_keep_matrix.astype(bool))
    max_per_side = reference_signals.shape[0]

    in_signals = []
    out_signals = []

    for data_idx in range(reference_signals.shape[1]):
        in_sig = reference_signals[ref_keep[:, data_idx], data_idx][:max_per_side]
        out_sig = reference_signals[~ref_keep[:, data_idx], data_idx][:max_per_side]
        if len(in_sig) == 0:
            in_sig = reference_signals[:, data_idx][:1]
        if len(out_sig) == 0:
            out_sig = reference_signals[:, data_idx][:1]
        in_signals.append(in_sig)
        out_signals.append(out_sig)

    in_signals = torch.stack(in_signals)
    out_signals = torch.stack(out_signals)

    if score_aggr_cfg.offline:
        ref_signals = out_signals.transpose(0, 1)
    else:
        ref_signals = torch.cat((in_signals, out_signals), dim=1).transpose(0, 1)

    target_indices = np.arange(len(membership))
    population_indices = np.where(~membership)[0]
    mean_x = _trim_mean_tensor(ref_signals[:, target_indices], score_aggr_cfg.proportiontocut, dim=0)
    mean_z = _trim_mean_tensor(ref_signals[:, population_indices], score_aggr_cfg.proportiontocut, dim=0)

    eps = 1e-45
    if score_aggr_cfg.offline:
        offline_a = score_aggr_cfg.offline_a
        prob_ratio_x = target_signal[target_indices] / (
            (((1 + offline_a) / 2) * mean_x + (1 - offline_a) / 2).clamp_min(eps)
        )
        prob_ratio_z_rev = 1.0 / (
            target_signal[population_indices]
            / ((((1 + offline_a) / 2) * mean_z + (1 - offline_a) / 2).clamp_min(eps))
            + eps
        )
    else:
        prob_ratio_x = target_signal[target_indices] / mean_x.clamp_min(eps)
        prob_ratio_z_rev = 1.0 / (target_signal[population_indices] / mean_z.clamp_min(eps) + eps)

    final_scores = torch.outer(prob_ratio_x.float(), prob_ratio_z_rev.float())
    scores = (final_scores > score_aggr_cfg.gamma).float().mean(dim=1)

    valid_population_indices = population_indices[population_indices < scores.shape[0]]
    if len(valid_population_indices) > 0 and len(mean_z) > 1:
        scores[valid_population_indices] = (
            (scores[valid_population_indices] * len(mean_z)) - 1.0
        ) / (len(mean_z) - 1)
    return scores

def _trim_mean_tensor(values, proportiontocut: float, dim: int = 0):
    if proportiontocut <= 0.0 or values.shape[dim] <= 2:
        return values.mean(dim=dim)
    sorted_values, _ = torch.sort(values, dim=dim)
    cut = int(values.shape[dim] * proportiontocut)
    if cut == 0 or (2 * cut) >= values.shape[dim]:
        return values.mean(dim=dim)
    index = [slice(None)] * values.ndim
    index[dim] = slice(cut, values.shape[dim] - cut)
    return sorted_values[tuple(index)].mean(dim=dim)












