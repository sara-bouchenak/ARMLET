import time
from typing import Generator, Iterable, Sequence
from torch.nn import Module
import hydra

from fluke import DDict
from fluke.client import Client
from fluke.server import Server
from fluke.data import DataSplitter, FastDataLoader
from fluke.config import OptimizerConfigurator

from armlet.FL_pipeline.FL_algorithms import ArmletCentralizedFL, ArmletClient, ArmletServer


class DataSelectionClient(ArmletClient):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        val_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        data_selection: DDict,
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        **kwargs,
    ):
        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            val_set=val_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping)

        if data_selection is not None:
            self.data_selection = hydra.utils.instantiate(data_selection, train_set=train_set)
        else:
            self.data_selection = None

    def fit(self, override_local_epochs: int = 0) -> float:
        
        num_epochs = override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs

        if self.data_selection is not None:

            start_select_samples_time = time.time()
            self.data_selection.pre_selection(self._last_round, self.model, self.device, self.hyper_params.loss_fn)
            self.train_set = self.data_selection.select_samples(self._last_round, num_epochs)
            end_select_samples_time = time.time()

            data_selection_metrics = {
                "select_samples_time": end_select_samples_time - start_select_samples_time,
                "n_selected_samples": self.train_set.tensors[0].shape[0],
                "max_n_available_samples": self.data_selection.initial_train_set.tensors[0].shape[0],
            }

            self.notify(
                event="track_item",
                round=self._last_round,
                #client_id=self.index,
                item="data_selection_client_{}".format(self.index),
                value=data_selection_metrics,
            )

        start_fit_time = time.time()
        loss = super().fit(override_local_epochs=override_local_epochs)
        end_fit_time = time.time()

        if self.data_selection is not None:
            sample_training_time_per_epoch = (end_fit_time - start_fit_time) / (self.train_set.size * num_epochs)
            self.data_selection.after_fit(self.model, self.device, self.hyper_params.loss_fn, sample_training_time_per_epoch)

        return loss


class DataSelectionServer(ArmletServer):

    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader | None,
        val_set: FastDataLoader | None,
        clients: Sequence[Client],
        data_selection: DDict,
        weighted: bool = False,
        lr: float = 1,
        **kwargs,
    ):
        super().__init__(model, test_set, val_set, clients, weighted, lr, **kwargs)

        if data_selection is not None:
            self.data_selection = hydra.utils.instantiate(data_selection)
        else:
            self.data_selection = None

        if self.data_selection is not None:
            self.data_selection.setup_hooks(model)

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True, **kwargs) -> None:
        if self.data_selection is not None:
            self.data_selection.before_fit(self.clients)
        super().fit(n_rounds, eligible_perc, finalize, **kwargs)

    def get_eligible_clients(self, eligible_perc: float) -> Sequence[Client]:

        if eligible_perc != 1.0 and self.data_selection is not None:

            start_select_clients_time = time.time()
            selected_clients = self.data_selection.select_clients(self.clients, eligible_perc)
            end_select_clients_time = time.time()

            client_selection_metrics = {
                "n_selected_clients": len(selected_clients),
                "select_clients_time": end_select_clients_time - start_select_clients_time,
            }

            self.notify(
                event="track_item",
                round=self.rounds+1,
                item="client_selection",
                value=client_selection_metrics,
            )

        else:
            selected_clients = super().get_eligible_clients(eligible_perc)

        return selected_clients

    def broadcast_model(self, eligible: Sequence[Client]) -> None:
        if self.data_selection is not None:
            self.data_selection.before_clients_local_updates(eligible, self.rounds)
        return super().broadcast_model(eligible)

    def receive_client_models(self, eligible: Sequence[Client], state_dict: bool = True) -> Generator[Module, None, None]:
        if self.data_selection is not None:
            self.data_selection.after_clients_local_updates(eligible, self.rounds)
        return super().receive_client_models(eligible, state_dict)

    def aggregate(self, eligible: Sequence[Client], client_models: Iterable[Module]) -> None:
        super().aggregate(eligible, client_models)

        if self.data_selection is not None:

            self.data_selection.after_aggregation(self.model, eligible, client_models)

            data_selection_metrics = self.data_selection.compute_other_metrics()
            if data_selection_metrics != {}:
                self.notify(
                    event="track_item",
                    round=self.rounds+1,
                    item="data_selection_server",
                    value=data_selection_metrics,
                )


class DataSelectionCentralizedFL(ArmletCentralizedFL):

    def __init__(self, n_clients: int, data_splitter: DataSplitter, hyperparameters: DDict, val_data: dict, **kwargs):
        hyper_params = DDict(hyperparameters)
        if "client" in hyper_params.data_selection.keys():
            hyper_params.client.data_selection = hyper_params.data_selection.client
        else:
            hyper_params.client.data_selection = None
        if "server" in hyper_params.data_selection.keys():
            hyper_params.server.data_selection = hyper_params.data_selection.server
        else:
            hyper_params.server.data_selection = None
        super().__init__(hyper_params, n_clients, data_splitter, val_data, **kwargs)

    def get_client_class(self) -> type[Client]:
        return DataSelectionClient

    def get_server_class(self) -> type[Server]:
        return DataSelectionServer
