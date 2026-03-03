import os

from torch.nn import Module

from fluke.utils.log import Log


class ArmletLog(Log):

    def __init__(self, json_log_dir: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.json_log_dir = json_log_dir

    def end_fit(self, round: int, client_id: int, model: Module, loss: float, **kwargs) -> None:
        loss_dict = {"training_loss": loss}
        self.tracker.add(perf_type="post-fit", metrics=loss_dict, round=round, client_id=client_id)
        return super().end_fit(round, client_id, model, loss, **kwargs)

    def close(self) -> None:
        if self.json_log_dir is not None:
            results_path = os.path.join(self.json_log_dir, "results.json")
            self.save(results_path)
        return super().close()
