from abc import ABC, abstractmethod
import torch
import hydra
import copy

from fluke.data import FastDataLoader
from fluke.utils import clear_cuda_cache


class MembershipInferenceAttack(ABC):

    def __init__(
        self,
        shadow_data,
        model_cfg,
        optimizer_cfg,
        loss_fn,
        device,
        shadow_train,
    ):

        self.shadow_train_set = shadow_data["train"]
        self.shadow_val_set = shadow_data["val"]
        self.shadow_test_set = shadow_data["test"]

        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.loss_func = loss_fn
        self.device = device

        self.shadow_train_cfg = shadow_train

    @abstractmethod
    def prepare_attack(self, attack_data_loader=None):
        raise NotImplementedError

    @abstractmethod
    def infer_membership(self, target_model, device, attack_data_loader):
        raise NotImplementedError

    def _train_shadow_model(
        self,
        train_dataloader,
        val_dataloader,
        n_epochs,
        print_every,
        model_label,
    ):

        print("### TRAIN {} ###".format(model_label))

        model = hydra.utils.instantiate(self.model_cfg)
        optimizer, scheduler = self.optimizer_cfg(model)

        model.to(self.device)

        train_losses = []
        val_losses = []

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")

        for epoch in range(n_epochs):

            epoch_loss_train = self._train_step(
                model,
                train_dataloader,
                optimizer,
                self.loss_func,
            )
            train_losses.append(epoch_loss_train)

            epoch_loss_val = self._eval_step(
                model,
                val_dataloader,
                self.loss_func,
            )
            val_losses.append(epoch_loss_val)

            if scheduler is not None:
                scheduler.step()

            if epoch_loss_val <= best_val_loss:
                best_val_loss = epoch_loss_val
                best_state = copy.deepcopy(model.state_dict())

            if (epoch+1) % print_every == 0:
                print("Epoch [{}/{}], Train Loss: {} | Val Loss: {}".format(
                    (epoch + 1), n_epochs, epoch_loss_train, epoch_loss_val
                ))

        model.load_state_dict(best_state)

        model.cpu()
        clear_cuda_cache()
        print()
        return model

    def _train_step(
        self,
        model,
        train_dataloader,
        optimizer,
        loss_func,
    ):
        model.train()
        epoch_loss = 0.0
        for batch in train_dataloader:
            X, y = batch[:2]
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if isinstance(train_dataloader, FastDataLoader):
            return epoch_loss / train_dataloader.tensors[0].shape[0]
        else:
            return epoch_loss / max(1, len(train_dataloader))

    def _eval_step(
        self,
        model,
        dataloader,
        loss_func,
    ):
        if dataloader is None:
            return float("inf")

        model.eval()
        epoch_loss = 0.0
        for batch in dataloader:
            X, y = batch[:2]
            X, y = X.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = model(X)
                epoch_loss += loss_func(y_hat, y).item()

        if isinstance(dataloader, FastDataLoader):
            return epoch_loss / dataloader.tensors[0].shape[0]
        else:
            return epoch_loss / max(1, len(dataloader))
