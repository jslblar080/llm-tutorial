import torch

from torch import device, Tensor
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
            print("inputs.shape \t\t targets.shape")
            for inputs, targets in train_dataloader:
                print(inputs.shape, "\t", targets.shape)
            print("\nValidation loader:")
            print("inputs.shape \t\t targets.shape")
            for inputs, targets in val_dataloader:
                print(inputs.shape, "\t", targets.shape)
        return train_dataloader, val_dataloader

    @staticmethod
    def _calc_batch_loss(
        input_batch: Tensor, target_batch: Tensor, device: device
    ) -> Tensor:
        inputs: Tensor = input_batch.to(device)
        targets: Tensor = target_batch.to(device)  # batch_size, context_length
        logits: Tensor = Config().gpt_model(
            inputs
        )  # batch_size, context_length, num_embeddings
        full_seq_loss_tensor = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        return full_seq_loss_tensor

    @classmethod
    def _calc_loader_loss(cls, dataloader: DataLoader[Tensor], device: device) -> float:
        total_loss = 0
        for inputs, targets in dataloader:
            total_loss += cls._calc_batch_loss(inputs, targets, device).item()
        return total_loss / len(dataloader)

    @classmethod
    def show_raw_loss(
        cls,
        train_loader: DataLoader[Tensor],
        val_loader: DataLoader[Tensor],
    ) -> None:
        config = Config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.gpt_model.to(device)
        with torch.no_grad():
            train_loss = cls._calc_loader_loss(train_loader, device)
            val_loss = cls._calc_loader_loss(val_loader, device)
        print("\nTraining loss:", train_loss)
        print("Validation loss:", val_loss)
