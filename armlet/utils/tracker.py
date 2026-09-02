from typing import Optional, Union, Literal

from fluke.evaluation import _compute_mean


class ArmletTracker():

    def __init__(self):

        self._performance = {
            "global": {},  # round -> evals
            "locals": {},  # round -> {client_id -> evals}
            "pre-fit": {},  # round -> {client_id -> evals}
            "post-fit": {},  # round -> {client_id -> evals}
            "comm": {0: 0},  # round -> fluke comm cost
            "mem": {},  # round -> fluke mem_usage
            "comp_cost": {},  # round -> armlet comp_cost
            "comm_cost": {},  # round -> armlet comm_cost
        }

    def add(
        self,
        perf_type: Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem", "comp_cost", "comm_cost"],
        metrics: dict[str, float] | float,
        round: int = 0,
        client_id: Optional[int] = None,
    ) -> None:
        """Add performance metrics for a specific type and client.

        Args:
            perf_type (Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem", "comp_cost", "comm_cost"]):
                The type of performance metrics to add.
            metrics (dict[str, float] | float): The performance metrics to add. If `perf_type`
                is "comm", this should be a single float value representing the communication cost.
            round (int, optional): The current round. Defaults to 0.
            client_id (int, optional): The client ID for local performance metrics. Defaults to
                None for global metrics.
        """
        if perf_type not in self._performance:
            raise ValueError(f"Unknown performance type: {perf_type}")

        if perf_type == "comm" or perf_type == "mem":
            if not isinstance(metrics, (float, int)):
                raise ValueError(f"Metrics for {perf_type} must be a float, got {type(metrics)}")
            if round not in self._performance[perf_type]:
                self._performance[perf_type][round] = metrics
            else:
                self._performance[perf_type][round] += metrics
        else:
            if round not in self._performance[perf_type]:
                self._performance[perf_type][round] = {}

            if perf_type == "global" or perf_type == "locals":
                self._performance[perf_type][round] = metrics

            elif perf_type == "comp_cost" or perf_type == "comm_cost":
                client_id_ = client_id if client_id is not None else "server"
                if client_id_ not in self._performance[perf_type][round].keys():
                    self._performance[perf_type][round][client_id_] = {}
                for key, value in metrics.items():
                    self._performance[perf_type][round][client_id_][key] = value

            else:
                self._performance[perf_type][round][client_id] = metrics

    def get(
        self,
        perf_type: Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem", "comp_cost", "comm_cost"],
        round: int,
    ) -> Union[dict, float]:
        """Get performance metrics for a specific type and round.

        Args:
            perf_type (Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem", "comp_cost", "comm_cost"]):
                The type of performance metrics to retrieve.
            round (int): The round for which to retrieve the metrics.

        Raises:
            ValueError: If the `perf_type` is unknown.

        Returns:
            Union[dict, float]: The performance metrics for the specified type and round.
                If `perf_type` is "comm" or "mem", returns a float; otherwise, returns a dict.
        """
        if perf_type not in self._performance:
            raise ValueError(f"Unknown performance type: {perf_type}")

        if round not in self._performance[perf_type]:
            return {} if perf_type not in ["comm", "mem"] else 0.0

        if perf_type in ["comm", "mem"]:
            return self._performance[perf_type][round]
        else:
            return self._performance[perf_type][round].copy()

    def __getitem__(self, item: str) -> dict:
        """Get performance metrics for a specific type.

        Args:
            item (str): The type of performance metrics to retrieve.

        Raises:
            ValueError: If the `item` is unknown.

        Returns:
            dict: The performance metrics for the specified type.
        """
        if item not in self._performance:
            raise ValueError(f"Unknown performance type: {item}")
        return self._performance[item].copy()

    def summary(
        self,
        perf_type: Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem", "comp_cost", "comm_cost"],
        round: int,
        include_round: bool = True,
        force_round: bool = True,
    ) -> Union[dict, float]:
        """Get the summary of the performance metrics for a specific type.

        Summary metrics are computed as the mean of the metrics for the specified type
        and round. If `perf_type` is "comm", the total communication cost is returned.
        If `perf_type` is "mem", the memory usage is returned.
        If `perf_type` is "global", the metrics are returned as they are.

        Args:
            perf_type (Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem", "comp_cost", "comm_cost"]):
                The type of performance metrics to retrieve.
            round (int): The round for which to compute the summary of the metrics.
            include_round (bool, optional): Whether to include the round number in the returned
                metrics. Defaults to `True`.
            force_round (bool, optional): If `True`, the method will return the metrics for the
                specified round if it exists, otherwise it will return the metrics for the
                latest round. Defaults to `False`.

        Raises:
            ValueError: If the `perf_type` is unknown or if there are no metrics for the specified
                type and round.

        Returns:
            Union[dict, float]: The summary performance metrics for the specified type.
        """
        if perf_type not in self._performance:
            raise ValueError(f"Unknown performance type: {perf_type}")

        if not self._performance[perf_type]:
            return {} if perf_type not in ["comm", "mem"] else 0.0

        if force_round:
            the_round = max(self._performance[perf_type].keys())
        elif round not in self._performance[perf_type]:
            return {}
        else:
            the_round = round

        if perf_type == "mem":
            return self._performance[perf_type][the_round]

        if perf_type == "comm":
            return sum(list(self._performance[perf_type].values()))

        if perf_type == "global":
            metrics = self._performance[perf_type][the_round].copy()
        else:
            metrics = _compute_mean(self._performance[perf_type][the_round])
            metrics["support"] = len(self._performance[perf_type][the_round])

        if include_round and metrics:
            metrics["round"] = the_round
        return metrics
