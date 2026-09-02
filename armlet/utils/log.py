import os
import time
import json
import torch

from fluke.utils.log import Log
from torch.nn import Module

from armlet.utils.tracker import ArmletTracker
from armlet.audit.auditor import OnlineAuditor


METRIC_TYPES_MAP = {
    "perf_global": "global",
    "perf_locals": "locals",
    "perf_prefit": "pre-fit",
    "perf_postfit": "post-fit",
    "comp_costs": "comp_cost",
    "comm_costs": "comm_cost"
}


class ArmletLog(Log):

    def __init__(
        self,
        log_dir: str | None = None,
        comm_costs_tracker : dict = {},
        metric_target : dict = {},
        auditor: OnlineAuditor | None = None,
        **kwargs,
    ):
        self.tracker = ArmletTracker()
        self.current_round: int = 0
        self.custom_fields: dict = {}
        self.log_dir = log_dir
        self.comm_costs_tracker_cfg = comm_costs_tracker
        self.metric_target_cfg = metric_target

        self.auditor = auditor

    def start_FL_process(self, **kwargs):
        self.start_FL_process_time = time.time()

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if "server_at_start_round" in self.comm_costs_tracker_cfg.keys() \
            and self.comm_costs_tracker_cfg["server_at_start_round"]:
            if self.comm_costs_tracker_cfg["track_every"] > 0 \
                and round % self.comm_costs_tracker_cfg["track_every"] == 0:
                self._track_communication_cost(
                    model=global_model,
                    round=round,
                    client_id=None,
                )

    def end_round(self, round: int) -> None:
        super().end_round(round)

        if self.auditor is not None:
            for metric_type in self.auditor.metric_types:
                metric_type_key = METRIC_TYPES_MAP[metric_type]
                if round in self.tracker[metric_type_key].keys():
                    self.auditor.add_metrics(
                        self.tracker[metric_type_key][round],
                        metric_type,
                        round,
                    )
            self.auditor.run_audit()

    def end_fit(self, round: int, client_id: int, model: Module, loss: float, **kwargs):
        if "clients_at_end_fit" in self.comm_costs_tracker_cfg.keys() \
            and self.comm_costs_tracker_cfg["clients_at_end_fit"]:
            if self.comm_costs_tracker_cfg["track_every"] > 0 \
                and round % self.comm_costs_tracker_cfg["track_every"] == 0:
                self._track_communication_cost(
                    model=model,
                    round=round,
                    client_id=client_id,
                )
        return super().end_fit(round, client_id, model, loss, **kwargs)

    def after_server_eval(self, metrics):
        for metric_name, metric_props in self.metric_target_cfg.items():
            if metric_props is not None and metric_name in metrics.keys():
                self._track_time_to_reach_metric_target(
                    metrics[metric_name], 
                    metric_name,
                    metric_props,
                )

    def finished(self, round: int) -> None:
        super().finished(round)
        self._track_computational_cost()

    def close(self) -> None:

        if self.log_dir is not None:
            results_path = os.path.join(self.log_dir, "results.json")
            self.save(results_path)

            if self.auditor is not None:
                self.auditor.save_audit_results()

        return super().close()

    def _track_communication_cost(
        self,
        model,
        round: int,
        client_id: int | None = None,
    ) -> None:
        comm_cost_metrics = {
            "model_size_in_bits": _calculate_model_size_in_bits(model),
            "n_model_params": _calculate_n_model_params(model),
        }
        self.tracker.add(
            perf_type="comm_cost",
            metrics=comm_cost_metrics,
            round=round,
            client_id=client_id,
        )

    def _track_computational_cost(self) -> None:
        end_FL_process_time = time.time()
        training_time = (end_FL_process_time - self.start_FL_process_time)
        #training_time_per_participant_per_round = training_time / (self.n_clients * eligible_perc * n_rounds)
        comp_cost_metrics = {
            "training_time": training_time,
        #    "training_time_per_participant_per_round": training_time_per_participant_per_round,
        }
        self.tracker.add(perf_type="comp_cost", metrics=comp_cost_metrics, round=-1, client_id=None)

    def _track_time_to_reach_metric_target(self, metric, metric_name, metric_props):

        if "threshold" not in metric_props.keys() and metric_props["threshold"] is None:
            return
        if "objective" not in metric_props.keys() and metric_props["objective"] is None:
            return

        if metric_props["objective"] == "above":
            if metric < metric_props["threshold"]:
                return
        elif metric_props["objective"] == "below":
            if metric > metric_props["threshold"]:
                return
        else:
            return

        target_metric_time = time.time()
        time_to_reach_target = target_metric_time - self.start_FL_process_time
        self.tracker.add(
            perf_type="comp_cost",
            metrics={"time_to_reach_{}_target".format(metric_name): time_to_reach_target},
            round=-1,
            client_id=None,
        )
        self.metric_target_cfg[metric_name] = None

    def save(self, path: str) -> None:
        """Save the logger's history to a JSON file.

        Args:
            path (str): The path to the JSON file.
        """
        json_to_save = {
            "perf_global": self.tracker["global"],
            "perf_locals": self.tracker["locals"],
            "perf_prefit": self.tracker["pre-fit"],
            "perf_postfit": self.tracker["post-fit"],
            "comp_costs": self.tracker["comp_cost"],
            "comm_costs": self.tracker["comm_cost"],
        }

        if self.custom_fields != {}:
            json_to_save["custom_fields"] = self.custom_fields

        with open(path, "w") as f:
            json.dump(json_to_save, f, indent=4)


def _calculate_model_size_in_bits(model):
    model_size = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            model_size += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            model_size += param.numel() * torch.iinfo(param.data.dtype).bits
    return model_size

def _calculate_n_model_params(model):
    return sum(param.numel() for param in model.parameters())
