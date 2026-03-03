from abc import ABC, abstractmethod
from typing import Sequence, Iterable
import numpy as np
import torch

from fluke.data import FastDataLoader
from fluke.client import Client


class DataSelectionClientMethod(ABC):

    def __init__(self, train_set: FastDataLoader):
        self.initial_train_set = train_set
        self.initial_data_size = train_set.max_size

    @abstractmethod
    def select_samples(self, round: int, num_epochs: int) -> FastDataLoader:
        raise NotImplementedError

    def after_fit(self, model, device, loss_fn, sample_training_time_per_epoch: float):
        pass

    def pre_selection(self, round, model, device, loss_fn):
        pass


class DataSelectionServerMethod(ABC):

    def __init__(self):
        pass

    def setup_hooks(self, model):
        pass

    # Random selection by default
    def select_clients(self, clients: Sequence[Client], eligible_perc: float) -> Sequence[Client]:
        n_selected_clients = max(1, int(len(clients) * eligible_perc))
        selected_clients = np.random.choice(clients, n_selected_clients, replace=False)
        return selected_clients

    def before_fit(self, clients: Sequence[Client]):
        pass

    def before_clients_local_updates(self, participants: Sequence[Client], round: int):
        pass

    def after_clients_local_updates(self, participants: Sequence[Client], round: int):
        pass

    def after_aggregation(
        self,
        server_model: torch.nn.Module,
        participants: Sequence[Client],
        clients_model: Iterable[torch.nn.Module],
    ):
        pass

    def compute_other_metrics(self) -> dict:
        return {}
