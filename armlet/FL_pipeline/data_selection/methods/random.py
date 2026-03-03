import numpy as np

from fluke.data import FastDataLoader

from armlet.FL_pipeline.data_selection.methods import DataSelectionClientMethod


class RandomDataSelection(DataSelectionClientMethod):

    def __init__(self, train_set: FastDataLoader, sampling_perc: float = 1.0, select_every: int = 1):
        super().__init__(train_set)

        self.last_train_set = train_set

        if sampling_perc > 1.0 or sampling_perc <= 0.0:
            raise ValueError("percentage must be in (0, 1]")
        self.sampling_perc = sampling_perc

        self.select_every = select_every

    def select_samples(self, round: int, num_epochs: int) -> FastDataLoader:
        if (round - 1) % self.select_every == 0:
            self.last_train_set = self._select_samples()
        return self.last_train_set

    def _select_samples(self) -> FastDataLoader:
        new_size = max(int(self.initial_data_size * self.sampling_perc), 1)
        sample_indices = np.random.choice(self.initial_data_size, new_size, replace=False)
        tensors = [t[sample_indices] for t in self.initial_train_set.tensors]

        selected_data = FastDataLoader(
            tensors[0],
            tensors[1],
            num_labels = self.initial_train_set.num_labels,
            batch_size = self.initial_train_set.batch_size,
            shuffle = self.initial_train_set.shuffle,
            transforms = self.initial_train_set.transforms,
            percentage = 1.0,
            skip_singleton = self.initial_train_set.skip_singleton,
            single_batch = self.initial_train_set.single_batch,
        )
        return selected_data
