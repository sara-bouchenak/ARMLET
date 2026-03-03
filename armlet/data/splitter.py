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


class ArmletDataSplitter:
    """Utility class for splitting the data across clients."""

    def __init__(
        self,
        data_dict: dict,
        dist_cfg: DDict,
        client_split: float = 0.0,
        client_val_split: float = 0.0,
        sampling_perc: float = 1.0,
        server_test: bool = True,
        server_test_union: bool = False,
        keep_test: bool = True,
        server_split: float = 0.0,
        server_val_split: float = 0.0,
        uniform_test: bool = False,
    ):

        assert 0 <= client_split <= 1, "client_split must be between 0 and 1."
        assert 0 <= client_val_split <= 1, "client_val_split must be between 0 and 1."
        assert 0 <= sampling_perc <= 1, "sampling_perc must be between 0 and 1."
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
        self.dist_cfg: DDict = dist_cfg
        self.client_split: float = client_split
        self.client_val_split: float = client_val_split
        self.sampling_perc: float = sampling_perc
        self.keep_test: bool = keep_test
        self.server_test: bool = server_test
        self.server_test_union: bool = server_test_union
        self.server_split: float = server_split
        self.server_val_split: float = server_val_split
        self.uniform_test: bool = uniform_test

    def assign(self, n_clients: int) -> dict:

        splitted_data = {}

        if self.server_test and self.keep_test:
            server_X, server_Y = self.data_dict["test"]
            clients_X, clients_Y = self.data_dict["train"]
            clients_X_tr, clients_X_te, clients_Y_tr, clients_Y_te = dataframe_safe_train_test_split(
                clients_X, clients_Y, test_size=self.client_split
            )
        elif not self.keep_test:
            X_tr, Y_tr = self.data_dict["train"]
            X_te, Y_te = self.data_dict["test"]
            # Merge and shuffle the data
            X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
            Y = pd.concat([Y_tr, Y_te], axis=0, ignore_index=True)
            idx = np.random.permutation(X.shape[0])
            X = X.loc[idx].reset_index(drop=True)
            Y = Y.loc[idx].reset_index(drop=True)
            # Split the data
            if self.server_test:
                clients_X, server_X, clients_Y, server_Y = dataframe_train_test_split(
                    X, Y, test_size=self.server_split
                )
            else:
                server_X, server_Y = None, None
                clients_X, clients_Y = X, Y
            clients_X_tr, clients_X_te, clients_Y_tr, clients_Y_te = dataframe_safe_train_test_split(
                clients_X, clients_Y, test_size=self.client_split
            )
        else:  # keep_test and not server_test
            server_X, server_Y = None, None
            clients_X_tr, clients_Y_tr = self.data_dict["train"]
            clients_X_te, clients_Y_te = self.data_dict["test"]

        assignments_tr, assignments_te = hydra.utils.instantiate(
            self.dist_cfg,
            X_train=clients_X_tr,
            y_train=clients_Y_tr,
            X_test=clients_X_te if not self.uniform_test else None,
            y_test=clients_Y_te if not self.uniform_test else None,
            n=n_clients,
        )

        if clients_X_te is not None and clients_Y_te is not None and self.uniform_test:
            assignments_te, _ = self.iid(clients_X_te, clients_Y_te, None, None, n_clients)

        client_training_sets = {}
        client_test_sets = {}
        client_val_sets = {}

        for c in range(n_clients):

            X_tr_client = clients_X_tr.loc[assignments_tr[c]].reset_index(drop=True)
            Y_tr_client = clients_Y_tr.loc[assignments_tr[c]].reset_index(drop=True)
            client_training_sets["client_{}".format(c)] = (X_tr_client, Y_tr_client)

            if clients_X_te is not None and clients_Y_te is not None:

                X_te_client = clients_X_te.loc[assignments_te[c]].reset_index(drop=True)
                Y_te_client = clients_Y_te.loc[assignments_te[c]].reset_index(drop=True)

                if self.client_val_split > 0:
                    X_val_client, X_te_client, Y_val_client, Y_te_client = dataframe_train_test_split(
                        X_te_client, Y_te_client, train_size=self.client_val_split
                    )
                    client_test_sets["client_{}".format(c)] = (X_te_client, Y_te_client)
                    client_val_sets["client_{}".format(c)] = (X_val_client, Y_val_client)
                else:
                    client_test_sets["client_{}".format(c)] = (X_te_client, Y_te_client)
                    client_val_sets["client_{}".format(c)] = None

            else:
                client_test_sets["client_{}".format(c)] = None
                client_val_sets["client_{}".format(c)] = None

        if self.server_test_union:
            server_X = pd.concat([X_te_client for (X_te_client, _) in client_test_sets.values()], axis=0, ignore_index=True)
            server_Y = pd.concat([Y_te_client for (_, Y_te_client) in client_test_sets.values()], axis=0, ignore_index=True)
            if self.client_val_split > 0:
                server_X_val = pd.concat([X_te_client for (X_te_client, _) in client_val_sets.values()], axis=0, ignore_index=True)
                server_Y_val = pd.concat([Y_te_client for (_, Y_te_client) in client_val_sets.values()], axis=0, ignore_index=True)
            else:
                server_X_val = None
                server_Y_val = None
        elif self.server_test and self.server_val_split > 0:
            server_X_val, server_X, server_Y_val, server_Y = dataframe_train_test_split(
                server_X, server_Y, train_size=self.server_val_split
            )
        else:
            server_X_val = None
            server_Y_val = None

        splitted_data["clients_train"] = client_training_sets
        splitted_data["clients_test"] = client_test_sets
        splitted_data["clients_val"] = client_val_sets
        splitted_data["server_test"] = (server_X, server_Y) if (self.server_test or self.server_test_union) else None
        server_val_cond = (self.server_test and self.server_val_split > 0) or (self.server_test_union and self.client_val_split > 0)
        splitted_data["server_val"] = (server_X_val, server_Y_val) if server_val_cond else None

        return splitted_data

    @staticmethod
    def iid(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.DataFrame],
        n: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray] | None]:

        assert X_train.shape[0] >= n, "# of instances must be > than #clients"
        assert X_test is None or X_test.shape[0] >= n, "# of instances must be > than #clients"

        assignments = []
        for X in (X_train, X_test):
            if X is None:
                assignments.append(None)
                continue
            ex_client = X.shape[0] // n
            idx = np.random.permutation(X.shape[0])
            assignments.append([idx[range(ex_client * i, ex_client * (i + 1))] for i in range(n)])
            # Assign the remaining examples one to every client until the examples are finished
            if X.shape[0] % n > 0:
                for cid, eid in enumerate(range(ex_client * n, X.shape[0])):
                    assignments[-1][cid] = np.append(assignments[-1][cid], idx[eid])

        return assignments[0], assignments[1]

    @staticmethod
    def label_dirichlet_skew(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.DataFrame],
        n: int,
        alpha: float = 0.1,
        min_sample_per_class: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray] | None]:

        assert alpha > 0, "alpha must be > 0"

        label_arrays = {}
        for key, y in zip(["train", "test"], [y_train, y_test]):
            if y is not None:
                assert not y.isnull().any().any()
                label_arrays[key] = y.to_numpy()

        unique_classes = list(np.unique(label_arrays["train"]))

        # Sample Dirichlet proportion
        proportions = np.random.dirichlet([alpha]*n, size=len(unique_classes))

        assignments = {}
        for key, label_array in label_arrays.items():
            
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

            assignments[key] = client_indices

        for key, assignment in assignments.items():
            assignments[key] = [np.array(assign) for assign in assignment.values()]

        if "test" not in assignments.keys():
            assignments["test"] = None

        return tuple(assignments.values())

    @staticmethod
    def safe_label_dirichlet_skew(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.DataFrame],
        n: int,
        alpha: float = 0.1,
    ) -> tuple[list[np.ndarray], list[np.ndarray] | None]:

        assignments_tr, assignments_te = ArmletDataSplitter.label_dirichlet_skew(X_train, y_train, X_test, y_test, n, alpha)

        client_have_no_data = False
        for assignments in (assignments_tr, assignments_te):
            if assignments is not None:
                for client_id in range(n):
                    n_samples = len(assignments[client_id])
                    #print("client_{}".format(client_id), "n_samples: {}".format(n_samples))
                    if n_samples == 0:
                        client_have_no_data = True

        if client_have_no_data:
            assignments_tr, assignments_te = ArmletDataSplitter.label_dirichlet_skew(X_train, y_train, X_test, y_test, n, alpha, min_sample_per_class=1)

        return (assignments_tr, assignments_te)
