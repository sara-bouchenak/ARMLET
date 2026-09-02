from typing import Any, Sequence
from torch.nn import Module
from copy import deepcopy
import hydra
import uuid
import warnings

from fluke import FlukeENV, DDict
from fluke.client import Client
from fluke.server import Server
from fluke.algorithms import CentralizedFL
from fluke.data import DataSplitter, FastDataLoader
from fluke.evaluation import Evaluator
from fluke.config import OptimizerConfigurator
from torch.utils.data import DataLoader


class ArmletClient(Client):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader | DataLoader,
        test_set: FastDataLoader | DataLoader,
        val_set: FastDataLoader | None,
        other_data: dict[str, FastDataLoader | DataLoader | None],
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        model_cfg,
        attack_cfg,
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        persistency: bool = True,
        **kwargs,
    ):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs, fine_tuning_epochs, clipping, persistency, **kwargs)
        self.val_set = val_set

        self.attack_evaluator = None
        if attack_cfg is not None:
            shadow_data = {
                key.replace("shadow_clients_", ""): val
                for key, val in other_data.items()
                if "shadow_clients" in key
            }
            self.attack_evaluator = hydra.utils.instantiate(
                attack_cfg.eval,
                _recursive_= False,
                train_set=train_set,
                test_set=test_set,
                attack_cfg=attack_cfg.exclude("eval"),
                shadow_data=shadow_data,
                model_cfg=model_cfg,
                optimizer_cfg=optimizer_cfg,
                loss_fn=loss_fn,
                device=self.device,
            )

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        model = self.model

        if test_set is not None and model is not None:
            metrics = evaluator.evaluate(
                self._last_round, model, test_set, device=self.device, loss_fn=self.hyper_params.loss_fn
            )
        else:
            metrics = {}

        if self.attack_evaluator is not None and self.model is not None:
            attack_evaluation = self.attack_evaluator.evaluate(
                round=self._last_round,
                model=self.model,
                device=self.device,
            )
            metrics.update(attack_evaluation)

        return metrics


class ArmletServer(Server):

    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader | None,
        val_set:  FastDataLoader | None,
        other_data: dict[str, FastDataLoader | DataLoader | None],
        clients: Sequence[Client],
        model_cfg,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        attack_cfg,
        weighted: bool = False,
        lr: float = 1,
        **kwargs,
    ):
        super().__init__(model, test_set, clients, weighted, lr, **kwargs)
        self.val_set = val_set

        if "loss" in kwargs.keys():
            self.loss_fn = hydra.utils.instantiate(kwargs["loss"])
        else:
            self.loss_fn = None

        self.attack_evaluator = None
        if attack_cfg is not None:
            train_set = other_data["server_train"]
            shadow_data = {
                key.replace("shadow_server_", ""): val
                for key, val in other_data.items()
                if "shadow_server" in key
            }
            self.attack_evaluator = hydra.utils.instantiate(
                attack_cfg.eval,
                _recursive_= False,
                train_set=train_set,
                test_set=test_set,
                attack_cfg=attack_cfg.exclude("eval"),
                shadow_data=shadow_data,
                model_cfg=model_cfg,
                optimizer_cfg=optimizer_cfg,
                loss_fn=loss_fn,
                device=self.device,
            )

    def fit(
        self, n_rounds: int = 10,
        eligible_perc: float = 0.1,
        finalize: bool = True,
        **kwargs,
    ) -> None:
        self.notify("start_FL_process")
        super().fit(n_rounds, eligible_perc, finalize, **kwargs)

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:

        if test_set is not None:
            metrics = evaluator.evaluate(
                self.rounds + 1, self.model, test_set, loss_fn=self.loss_fn, device=self.device
            )
        else:
            metrics = {}
        self.notify("after_server_eval", metrics=metrics)

        if self.attack_evaluator is not None and self.model is not None:
            attack_evaluation = self.attack_evaluator.evaluate(
                round=self.rounds+1,
                model=self.model,
                device=self.device,
            )
            metrics.update(attack_evaluation)

        return metrics

    def load(self, path: str) -> dict[str, Any]:
        state = super().load(path)
        self.rounds +=1 # fix Fluke bug
        return state

class ArmletCentralizedFL(CentralizedFL):

    def __init__(
        self,
        hyperparameters: DDict | dict[str, Any],
        n_clients: int,
        data_splitter: DataSplitter,
        additional_data: dict,
        clients: list[Client] = None,
        server: Server = None,
        **kwargs
    ):

        self.clients_val = additional_data["clients_val"]
        self.server_val = additional_data["server_val"]

        self.other_server_data = {
            key: val
            for key, val in additional_data.items()
            if ("server" in key) and (key != "server_val")
        }

        self.other_clients_data = {
            key: val
            for key, val in additional_data.items()
            if ("clients" in key) and (key != "client_val")
        }

        hyper_params = hyperparameters

        if (clients is not None and server is None) or (clients is None and server is not None):
            raise ValueError("Both clients and server must be provided or neither of them.")

        self._id = str(uuid.uuid4().hex)
        FlukeENV().open_cache(self._id)

        if clients is not None:
            self.clients = clients
            self.n_clients = len(clients)
            if self.n_clients != n_clients:
                warnings.warn(
                    f"Number of clients provided ({self.n_clients}) is different from"
                    + f"the number of clients expected ({n_clients}). Overwriting "
                    + f"the number of clients to {self.n_clients}."
                )
            self.server = server
            model_name = "Unknown"
            if server.model is not None:
                model_name = server.model.__class__.__name__
            else:
                model_name = clients[0].model.__class__.__name__
            self.hyper_params = DDict(
                client=clients[0].hyper_params, server=server.hyper_params, model=model_name
            )

        else:
            if isinstance(hyper_params, dict):
                hyper_params = DDict(hyper_params)

            self.hyper_params = hyper_params
            self.n_clients = n_clients
            (clients_tr_data, clients_te_data), server_data = data_splitter.assign(
                n_clients, hyper_params.client.batch_size
            )
            # Federated model
            model = hydra.utils.instantiate(hyper_params.model)

            self.clients = self.init_clients(clients_tr_data, clients_te_data, hyper_params.client)
            self.server = self.init_server(model, server_data, hyper_params.server)

        for client in self.clients:
            client.set_channel(self.server.channel)

        if "resume" in kwargs.keys() and kwargs["resume"]:
            resume_cfg = kwargs["resume"]
            assert "path" in resume_cfg
            round = resume_cfg["round"] if "round" in resume_cfg.keys() else None
            self.load(resume_cfg["path"], round)

    def get_client_class(self) -> type[Client]:
        return ArmletClient

    def get_server_class(self) -> type[Server]:
        return ArmletServer

    def init_server(self, model: Any, data: FastDataLoader, config: DDict) -> Server:
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=self.hyper_params.client.optimizer,
            scheduler_cfg=self.hyper_params.client.scheduler,
        )
        loss = hydra.utils.instantiate(self.hyper_params.client.loss)
        attack_cfg = self.hyper_params.attack if "attack" in self.hyper_params.keys() else None
        server: Server = self.get_server_class()(
            model=model,
            test_set=data,
            val_set=self.server_val,
            other_data=self.other_server_data,
            clients=self.clients,
            model_cfg=self.hyper_params.model,
            optimizer_cfg=optimizer_cfg,
            loss_fn=deepcopy(loss),
            attack_cfg=attack_cfg,
            **config,
        )
        if FlukeENV().get_save_options()[0] is not None:
            server.attach(self)
        return server

    def init_clients(
        self,
        clients_tr_data: list[FastDataLoader],
        clients_te_data: list[FastDataLoader],
        config: DDict,
    ) -> Sequence[Client]:
        self._fix_opt_cfg(config.optimizer)
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=config.optimizer,
            scheduler_cfg=config.scheduler,
        )
        loss = hydra.utils.instantiate(config.loss)
        attack_cfg = self.hyper_params.attack if "attack" in self.hyper_params.keys() else None
        clients = [
            self.get_client_class()(
                index=i,
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                val_set=self.clients_val[i],
                other_data={key: val[i] for key, val in self.other_clients_data.items()},
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(loss),
                model_cfg=self.hyper_params.model,
                attack_cfg=attack_cfg,
                **config.exclude("optimizer", "loss", "batch_size", "scheduler"),
            )
            for i in range(self.n_clients)
        ]
        return clients
