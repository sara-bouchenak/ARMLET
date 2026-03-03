from fluke.data import FastDataLoader

from armlet.FL_pipeline.data_selection.methods import DataSelectionClientMethod


class FullDataSelection(DataSelectionClientMethod):

    def __init__(self, train_set: FastDataLoader):
        super().__init__(train_set)
        self.initial_train_set.percentage = 1.0

    def select_samples(self, round: int, num_epochs: int) -> FastDataLoader:
        return self.initial_train_set
