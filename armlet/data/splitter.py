"""
This module contains the data partitioning utilities.
"""

from typing import Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import hydra

from fluke import DDict
from fluke.data import DataSplitter, FastDataLoader, DummyDataContainer

from armlet.data.utils import dataframe_train_test_split, dataframe_safe_train_test_split


class ArmletDataSplitter:
    """Utility class for splitting the data across clients."""

    def __init__(
        self,
        data_dict: dict,
        distribution: DDict,
        client_split: float = 0.0,
        client_val_split: float = 0.0,
        server_test: bool = True,
        server_test_union: bool = False,
        keep_test: bool = True,
        server_split: float = 0.0,
        server_val_split: float = 0.0,
        uniform_test: bool = False,
    ):

        assert 0 <= client_split <= 1, "client_split must be between 0 and 1."
        assert 0 <= client_val_split <= 1, "client_val_split must be between 0 and 1."
        assert 0 <= server_split <= 1, "server_split must be between 0 and 1."
        assert 0 <= server_val_split <= 1, "server_val_split must be between 0 and 1."
        if not keep_test and server_test and server_split == 0.0:
            raise AssertionError(
                "server_split must be > 0.0 if server_test is True and keep_test is False."
            )
        if not server_test and client_split == 0.0:
            raise AssertionError("Either client_split > 0 or server_test = True must be true.")
        if server_test_union and server_test:
            raise AssertionError("server_test must be False if server_test_union is True.")
        if server_test_union and keep_test:
            raise AssertionError("keep_test must be False if server_test_union is True.")

        self.data_dict: dict = data_dict
        self.dist_cfg: DDict = distribution
        self.client_split: float = client_split
        self.client_val_split: float = client_val_split
        self.keep_test: bool = keep_test
        self.server_test: bool = server_test
        self.server_test_union: bool = server_test_union
        self.server_split: float = server_split
        self.server_val_split: float = server_val_split
        self.uniform_test: bool = uniform_test

    def assign(self, n_clients: int) -> dict:

        data = self._init_client_and_server_data(self.data_dict)

        splitted_data = {}

        assignments_tr = self._compute_train_assignments(n_clients, data["clients_train"])
        splitted_data["clients_train"] = self._assign_client_data(n_clients, data["clients_train"], assignments_tr)

        assignments_te = self._compute_test_val_assignments(n_clients, data["clients_test"])
        splitted_data["clients_test"] = self._assign_client_data(n_clients, data["clients_test"], assignments_te)

        assignments_val = self._compute_test_val_assignments(n_clients, data["clients_val"])
        splitted_data["clients_val"] = self._assign_client_data(n_clients, data["clients_val"], assignments_val)

        splitted_data["server_test"], splitted_data["server_val"] = self._assign_server_data(
            data["server_test"], data["clients_test"], data["clients_val"],
        )

        return splitted_data

    def _init_client_and_server_data(self, data_dict):

        if self.keep_test:
            clients_X_tr, clients_Y_tr = data_dict["train"]
            if not self.server_test:
                server_X, server_Y = None, None
                clients_X_te, clients_Y_te = data_dict["test"]
            else:
                server_X, server_Y = data_dict["test"]
                clients_X_tr, clients_X_te, clients_Y_tr, clients_Y_te = dataframe_safe_train_test_split(
                    clients_X_tr, clients_Y_tr, test_size=self.client_split,
                )

        else:

            clients_X_tr, clients_Y_tr = self._concat_and_shuffle(data_dict["train"], data_dict["test"])

            if self.server_test:
                clients_X_tr, server_X, clients_Y_tr, server_Y = dataframe_train_test_split(
                    clients_X_tr, clients_Y_tr, test_size=self.server_split,
                )
            else:
                server_X, server_Y = None, None

            clients_X_tr, clients_X_te, clients_Y_tr, clients_Y_te = dataframe_safe_train_test_split(
                clients_X_tr, clients_Y_tr, test_size=self.client_split,
            )

        if clients_X_te is not None and clients_Y_te is not None:
            clients_X_te, clients_X_val, clients_Y_te, clients_Y_val = dataframe_train_test_split(
                clients_X_te, clients_Y_te, test_size=self.client_val_split,
            )
        else:
            clients_X_val, clients_Y_val = None, None

        data_dict = {}
        data_dict["clients_train"] = (clients_X_tr, clients_Y_tr)
        data_dict["clients_test"] = (clients_X_te, clients_Y_te)
        data_dict["clients_val"] = (clients_X_val, clients_Y_val)
        data_dict["server_test"] = (server_X, server_Y)

        return data_dict

    def _concat_and_shuffle(self, data_dict_train, data_dict_test) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_tr, Y_tr = data_dict_train
        X_te, Y_te = data_dict_test
        X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
        Y = pd.concat([Y_tr, Y_te], axis=0, ignore_index=True)
        idx = np.random.permutation(X.shape[0])
        X = X.loc[idx].reset_index(drop=True)
        Y = Y.loc[idx].reset_index(drop=True)
        return X, Y

    def _compute_train_assignments(self, n_clients, clients_tr_data):
        clients_X_tr, clients_Y_tr = clients_tr_data
        assignments_tr = hydra.utils.instantiate(
            self.dist_cfg, X=clients_X_tr, y=clients_Y_tr, n=n_clients,
        )
        return assignments_tr

    def _compute_test_val_assignments(self, n_clients, clients_data):
        clients_X, clients_Y = clients_data
        if clients_X is not None and clients_Y is not None:
            if self.uniform_test:
                assignments = self.iid(X=clients_X, n=n_clients)
            else:
                assignments = hydra.utils.instantiate(
                    self.dist_cfg, X=clients_X, y=clients_Y, n=n_clients,
                )
        else:
            assignments = None
        return assignments

    def _assign_client_data(self, n_clients, clients_data, assignments):
        clients_X, clients_Y = clients_data
        client_sets = {}
        for c in range(n_clients):
            if assignments is not None:
                X_client = clients_X.loc[assignments[c]].reset_index(drop=True)
                Y_client = clients_Y.loc[assignments[c]].reset_index(drop=True)
                client_set = (X_client, Y_client)
            else:
                client_set = None
            client_sets["client_{}".format(c)] = client_set
        return client_sets

    def _assign_server_data(self, server_data, clients_te_data, clients_val_data):

        if self.server_test_union:
            server_te_data = clients_te_data
            if self.client_val_split > 0:
                server_val_data = clients_val_data
            else:
                server_val_data = None
        else:
            server_X, server_Y = server_data
            if server_X is not None and server_Y is not None:
                server_X, server_X_val, server_Y, server_Y_val = dataframe_train_test_split(
                    server_X, server_Y, test_size=self.server_val_split,
                )
                server_te_data = (server_X, server_Y)
                if (server_X_val is not None and server_Y_val is not None):
                    server_val_data = (server_X_val, server_Y_val)  
                else:
                    server_val_data = None
            else:
                server_te_data, server_val_data = None, None

        return server_te_data, server_val_data

    @staticmethod
    def iid(X: pd.DataFrame, n: int, **kwargs) -> list[np.ndarray]:

        assert X.shape[0] >= n, "# of instances must be > than #clients"

        ex_client = X.shape[0] // n
        idx = np.random.permutation(X.shape[0])
        assignments = [idx[range(ex_client * i, ex_client * (i + 1))] for i in range(n)]
        # Assign the remaining examples one to every client until the examples are finished
        if X.shape[0] % n > 0:
            for cid, eid in enumerate(range(ex_client * n, X.shape[0])):
                assignments[cid] = np.append(assignments[cid], idx[eid])

        return assignments

    @staticmethod
    def label_dirichlet_skew(
        y: pd.DataFrame,
        n: int,
        alpha: float = 0.1,
        min_sample_per_class: int = 0,
        seed = None,
        **kwargs,
    ) -> list[np.ndarray]:

        assert alpha > 0, "alpha must be > 0"

        assert not y.isnull().any().any()
        label_array = y.to_numpy()

        unique_classes = list(np.unique(label_array))

        # Sample Dirichlet proportion
        if seed is not None:
            rng = np.random.default_rng(seed)
            proportions = rng.dirichlet([alpha]*n, size=len(unique_classes))
        else:
            proportions = np.random.dirichlet([alpha]*n, size=len(unique_classes))

        # Initialize client allocations
        client_indices = defaultdict(list)

        # Distribute data using proportions
        for class_id, c in enumerate(unique_classes):
            indices_class_c = np.where(label_array == c)[0].tolist()
            np.random.shuffle(indices_class_c)

            # Pre-allocate the minimum number of sample to each client
            if min_sample_per_class > 0:
                tot_min_sample = min_sample_per_class * n
                indices_to_pre_allocate = indices_class_c[:tot_min_sample]
                indices_to_pre_allocate = np.split(np.array(indices_to_pre_allocate), n)
                indices_class_c = indices_class_c[tot_min_sample:]

                for client_id in range(n):
                    client_indices[client_id].extend(indices_to_pre_allocate[client_id])

            normalized_proportion = proportions[class_id] / sum(proportions[class_id])
            cumsum_division_numbers = np.cumsum(normalized_proportion) * len(indices_class_c)
            indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]
            split_client_indices = np.split(indices_class_c, indices_on_which_split)

            for client_id in range(n):
                client_indices[client_id].extend(split_client_indices[client_id])

        for client_id in range(n):
            np.random.shuffle(client_indices[client_id])

        assignments = [np.array(assign) for assign in client_indices.values()]

        return assignments

    @staticmethod
    def safe_label_dirichlet_skew(
        n: int,
        min_sample_per_class: int,
        **kwargs,
    ) -> list[np.ndarray]:

        assignments = ArmletDataSplitter.label_dirichlet_skew(n=n, **kwargs)

        client_have_no_data = False

        for client_id in range(n):
            n_samples = len(assignments[client_id])
            #print("client_{}".format(client_id), "n_samples: {}".format(n_samples))
            if n_samples == 0:
                client_have_no_data = True

        if client_have_no_data:
            assignments = ArmletDataSplitter.label_dirichlet_skew(
                n=n,
                min_sample_per_class=min_sample_per_class,
                **kwargs,
            )

        return assignments


class DummyDataSplitter(DataSplitter):

    def assign(
        self, n_clients: int, batch_size: int = 32
    ) -> tuple[tuple[list[FastDataLoader], Optional[list[FastDataLoader]]], FastDataLoader]:
        assert isinstance(self.data_container, DummyDataContainer)
        assert n_clients <= len(self.data_container.clients_tr), (
            "n_clients must be <= of " + "the number of clients in the `DummyDataContainer`."
        )
        client_ids = range(n_clients)
        clients_tr = [self.data_container.clients_tr[i] for i in client_ids]
        clients_te = [self.data_container.clients_te[i] for i in client_ids]
        return (clients_tr, clients_te), self.data_container.server_data
