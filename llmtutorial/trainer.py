import torch

from torch import Tensor
from torch.utils.data import DataLoader
from .config import Config
from .text_processor import TextProcessor


class Trainer:

    @staticmethod
    def create_dataloader(
        text_data: str, verbose=False
    ) -> tuple[DataLoader[Tensor], DataLoader[Tensor]]:
        config = Config()
        split_idx = int(config.train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]
        torch.manual_seed(config.seed_num)
        train_token_ids = TextProcessor.tokenize(train_data, id_end=True, pair=False)
        Config().dataset = train_token_ids
        train_dataset = Config().dataset
        train_dataloader: DataLoader[Tensor] = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        val_token_ids = TextProcessor.tokenize(val_data, id_end=True, pair=False)
        Config().dataset = val_token_ids
        val_dataset = Config().dataset
        val_dataloader: DataLoader[Tensor] = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        if verbose:
            print("\nTrain loader:")
            for inputs, targets in train_dataloader:
                print(inputs.shape, targets.shape)
            print("\nValidation loader:")
            for inputs, targets in val_dataloader:
                print(inputs.shape, targets.shape)
        return train_dataloader, val_dataloader
