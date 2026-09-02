import torch
import hydra
import copy

from torch.utils.data.dataset import TensorDataset

from fluke.utils import clear_cuda_cache

from armlet.attack.mia import MembershipInferenceAttack
from armlet.attack.metrics import membership_scores


class ShokriMIA(MembershipInferenceAttack):

    def __init__(
        self,
        shadow_data,
        model_cfg,
        optimizer_cfg,
        loss_fn,
        device,
        shadow_train,
        attack_train,
    ):

        super().__init__(
            shadow_data,
            model_cfg,
            optimizer_cfg,
            loss_fn,
            device,
            shadow_train,
        )

        self.attack_train_cfg = attack_train

    def prepare_attack(self, attack_data_loader=None):

        shadow_model = self._train_shadow_model(
            train_dataloader=self.shadow_train_set,
            val_dataloader=self.shadow_val_set,
            n_epochs=self.shadow_train_cfg.n_epochs,
            print_every=self.shadow_train_cfg.print_every,
            model_label="SHADOW MODEL",
        )

        attack_train_dl, attack_val_dl = self._construct_attack_dataset(
            shadow_model,
            self.attack_train_cfg,
        )

        self.attack_model = self._train_attack_model(
            attack_train_dl,
            attack_val_dl,
            self.attack_train_cfg,
        )

    def _construct_attack_dataset(
            self,
            shadow_model,
            attack_train_cfg,
        ):

        X_member = self._compute_X_attack(shadow_model, self.shadow_train_set)
        X_non_member = self._compute_X_attack(shadow_model, self.shadow_test_set)
        X_attack = torch.cat([X_member, X_non_member])

        y_member = torch.ones(X_member.shape[0], 1)
        y_non_member = torch.zeros(X_non_member.shape[0], 1)
        y_attack = torch.cat([y_member, y_non_member])

        attack_dataset = TensorDataset(X_attack, y_attack)

        val_size = int(len(attack_dataset) * attack_train_cfg.val_split)
        train_size = len(attack_dataset) - val_size
        train_data, val_data = torch.utils.data.random_split(attack_dataset, [train_size, val_size])

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=attack_train_cfg.batch_size,
            shuffle=True,
            num_workers=attack_train_cfg.num_workers,
        )

        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=attack_train_cfg.batch_size,
            shuffle=False,
            num_workers=attack_train_cfg.num_workers,
        )

        return train_dataloader, val_dataloader

    def _compute_X_attack(self, model, dataloader):
        X_attack_list = []

        model.to(self.device)
        model.eval()

        for batch in dataloader:
            X, y = batch[:2]
            X, y = X.to(self.device), y.to(self.device)
            with torch.no_grad():
                logits = model(X)

            X_attack = self._concat_label_and_probs(y, logits)
            X_attack_list.append(X_attack)

        model.cpu()
        clear_cuda_cache()

        return torch.cat(X_attack_list)

    def _concat_label_and_probs(self, y, logits):

        probs = _model_probabilities(logits)
        label_features = _label_features(y, probs.shape[-1]).to(probs.device)

        if self.attack_train_cfg.top_k:
            top_k_probs, _ = torch.topk(probs, self.attack_train_cfg.n_class_top_k, dim=1)
            out = torch.cat([label_features, top_k_probs.cpu()], dim=-1)
        else:
            out = torch.cat([label_features, probs], dim=-1)
        return out.detach().cpu()

    def _train_attack_model(
            self,
            train_dataloader,
            val_dataloader,
            attack_train_cfg,
        ):

        print("### TRAIN ATTACK MODEL ###")

        input_size = train_dataloader.dataset[0][0].numel()

        model = hydra.utils.instantiate(attack_train_cfg.model, input_size=input_size)
        loss = hydra.utils.instantiate(attack_train_cfg.loss)
        optimizer = hydra.utils.instantiate(attack_train_cfg.optimizer, params=model.parameters())

        model.to(self.device)

        train_losses = []
        val_losses = []

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")

        for epoch in range(attack_train_cfg.n_epochs):

            epoch_loss_train = self._train_step(
                model,
                train_dataloader,
                optimizer,
                loss,
            )
            train_losses.append(epoch_loss_train)

            epoch_loss_val = self._eval_step(
                model,
                val_dataloader,
                loss,
            )
            val_losses.append(epoch_loss_val)

            if epoch_loss_val <= best_val_loss:
                best_val_loss = epoch_loss_val
                best_state = copy.deepcopy(model.state_dict())

            if (epoch+1) % attack_train_cfg.print_every == 0:
                print("Epoch [{}/{}], Train Loss: {} | Val Loss: {}".format(
                    (epoch + 1), attack_train_cfg.n_epochs, epoch_loss_train, epoch_loss_val
                ))

        model.load_state_dict(best_state)

        model.cpu()
        clear_cuda_cache()
        print()
        return model

    def infer_membership(self, target_model, device, attack_data_loader):

        model_device = torch.device("cpu")
        if next(target_model.parameters(), None) is not None:
            model_device = next(target_model.parameters()).device
        target_model.eval()
        target_model.to(device)

        all_scores = []
        all_targets = []
        all_logits = []

        for (X, y, attack_y) in attack_data_loader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_hat = target_model(X)
            all_logits.append(y_hat.detach().cpu().reshape(y_hat.shape[0], -1))
            score = membership_scores(self._inference_with_attack_model(y, y_hat))
            all_scores.append(score.detach().cpu().reshape(-1))
            all_targets.append(attack_y.detach().cpu().reshape(-1))

        y_score = torch.cat(all_scores).numpy()
        y_true = torch.cat(all_targets).numpy().astype(int)
        logits = torch.cat(all_logits).numpy()

        target_model.to(model_device)
        clear_cuda_cache()
        return y_score, y_true, logits

    def _inference_with_attack_model(self, y, y_hat):
        X_attack = self._concat_label_and_probs(y, y_hat).to(self.device)

        self.attack_model.eval()
        self.attack_model.to(self.device)

        with torch.no_grad():
            out = self.attack_model(X_attack)

        self.attack_model.cpu()
        clear_cuda_cache()

        return out.detach().cpu()


def _model_probabilities(logits):
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.shape[-1] == 1:
        if logits.is_floating_point() and torch.all((logits >= 0) & (logits <= 1)):
            pos = logits
        else:
            pos = torch.sigmoid(logits)
        return torch.cat([1 - pos, pos], dim=-1)
    return torch.nn.functional.softmax(logits, dim=-1)

def _label_features(y, num_classes: int):
    y = y.squeeze(-1).long()
    if num_classes <= 2:
        return y.float().unsqueeze(-1)
    return torch.nn.functional.one_hot(y, num_classes=num_classes).float()
