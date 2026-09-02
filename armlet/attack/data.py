import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader
from fluke.data import FastDataLoader


def construct_attack_test_dataloader(train_set, test_set):
        if isinstance(train_set, FastDataLoader):
            return _construct_attack_test_fast_dataloader(
                train_set,
                test_set,
            )
        elif isinstance(train_set, DataLoader):
            return _construct_attack_test_torch_dataloader(
                train_set,
                test_set,
            )
        else:
            raise ValueError("The dataset type is not compatible with the attack evaluator!")

def _construct_attack_test_fast_dataloader(train_set, test_set):

    train_tensors = train_set.tensors[:2]
    test_tensors = test_set.tensors[:2]

    new_size = min(train_tensors[0].shape[0], test_tensors[0].shape[0])

    if new_size < train_tensors[0].shape[0]:
        sample_indices = np.random.choice(train_tensors[0].shape[0], new_size, replace=False)
        train_tensors = [t[sample_indices] for t in train_tensors]

    if new_size < test_tensors[0].shape[0]:
        sample_indices = np.random.choice(test_tensors[0].shape[0], new_size, replace=False)
        test_tensors = [t[sample_indices] for t in test_tensors]

    tensors = [torch.concat([t_train, t_test]) for t_train, t_test in zip(train_tensors, test_tensors)]
    attack_y = torch.concat([
        torch.ones(train_tensors[0].shape[0]),
        torch.zeros(test_tensors[0].shape[0]),
    ]).unsqueeze(-1)

    return FastDataLoader(
        tensors[0],
        tensors[1],
        attack_y,
        num_labels=test_set.num_labels,
        batch_size=test_set.batch_size,
        shuffle=test_set.shuffle,
        transforms=test_set.transforms,
        percentage=1.0,
        skip_singleton=test_set.skip_singleton,
        single_batch=test_set.single_batch,
    )

def _construct_attack_test_torch_dataloader(
    train_set,
    test_set,
):
    train_dataset = train_set.dataset
    test_dataset = test_set.dataset

    new_size = min(
        len(train_dataset),
        len(test_dataset),
    )

    if new_size < len(train_dataset):
        train_indices = torch.randperm(len(train_dataset))[:new_size].tolist()
    else:
        train_indices = list(range(new_size))

    if new_size < len(test_dataset):
        test_indices = torch.randperm(len(test_dataset))[:new_size].tolist()
    else:
        test_indices = list(range(new_size))

    member_dataset = _MembershipDataset(
        dataset=train_dataset,
        indices=train_indices,
        membership=1.0,
    )

    non_member_dataset = _MembershipDataset(
        dataset=test_dataset,
        indices=test_indices,
        membership=0.0,
    )

    attack_dataset = ConcatDataset([
        member_dataset,
        non_member_dataset,
    ])

    return DataLoader(
        dataset=attack_dataset,
        batch_size=test_set.batch_size,
        shuffle=False,
        num_workers=test_set.num_workers,
        pin_memory=test_set.pin_memory,
        drop_last=False,
    )


class _MembershipDataset(Dataset):
    def __init__(
        self,
        dataset,
        indices,
        membership: float,
    ):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.membership = membership

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        X, y = self.dataset[original_index]
        attack_y = torch.tensor(
            [self.membership],
            dtype=torch.float32,
        )
        return X, y, attack_y
