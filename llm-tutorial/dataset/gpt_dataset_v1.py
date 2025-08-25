import torch

from base_dataset import BaseDataset
from torch import Tensor


class GPTDatasetV1(BaseDataset):

    _input_ids = []
    _target_ids = []

    def __init__(self, token_ids: list[int], max_length: int, stride: int) -> None:
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self._input_ids.append(torch.tensor(input_chunk))
            self._target_ids.append(torch.tensor(target_chunk))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._input_ids[index], self._target_ids[index]

    def __len__(self) -> int:
        return len(self._input_ids)
