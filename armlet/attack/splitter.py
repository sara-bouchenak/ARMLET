from fluke import DDict

from armlet.data.splitter import ArmletDataSplitter
from armlet.data.utils import dataframe_safe_train_test_split


class ShadowDataSplitter(ArmletDataSplitter):
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
        shadow_perc: float = 0.5,
    ):
        super().__init__(
            data_dict,
            distribution,
            client_split,
            client_val_split,
            server_test,
            server_test_union,
            keep_test,
            server_split,
            server_val_split,
            uniform_test,
        )
        assert 0 <= shadow_perc <= 1, "shadow_perc must be between 0 and 1."
        self.shadow_perc: float = shadow_perc

    def assign(self, n_clients: int) -> dict:

        new_data_dict = {}
        self.data_dict_shadow = {}
        for key, val in self.data_dict.items():
            X, y = val
            X, X_shadow, y, y_shadow = dataframe_safe_train_test_split(
                X, y, test_size=self.shadow_perc,
            )
            new_data_dict[key] = (X, y)
            self.data_dict_shadow[key] = (X_shadow, y_shadow)
        self.data_dict = new_data_dict

        splitted_data = self.assign_FL_data(n_clients)

        shadow_splitted_data = self.assign_shadow(n_clients)

        for key, val in shadow_splitted_data.items():
            splitted_data[key] = val

        return splitted_data

    def assign_FL_data(self, n_clients: int) -> dict:

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

        # server train data is the union of the clients train data
        splitted_data["server_train"] = data["clients_train"]

        return splitted_data

    def assign_shadow(self, n_clients: int) -> dict:

        data = self._init_client_and_server_data(self.data_dict_shadow)

        splitted_data = {}

        assignments_tr = self._compute_train_assignments(n_clients, data["clients_train"])
        splitted_data["shadow_clients_train"] = self._assign_client_data(n_clients, data["clients_train"], assignments_tr)

        assignments_te = self._compute_test_val_assignments(n_clients, data["clients_test"])
        splitted_data["shadow_clients_test"] = self._assign_client_data(n_clients, data["clients_test"], assignments_te)

        assignments_val = self._compute_test_val_assignments(n_clients, data["clients_val"])
        splitted_data["shadow_clients_val"] = self._assign_client_data(n_clients, data["clients_val"], assignments_val)

        splitted_data["shadow_server_test"], splitted_data["shadow_server_val"] = self._assign_server_data(
            data["server_test"], data["clients_test"], data["clients_val"],
        )

        # shadow server train data is the union of the shadow clients train data
        splitted_data["shadow_server_train"] = data["clients_train"]

        return splitted_data
