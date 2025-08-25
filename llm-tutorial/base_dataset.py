from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self) -> None:
        pass

    def __getitem__(self, index: int) -> None:
        raise NotImplementedError(
            f'{type(self).__name__} is missing the implementation of "__getitem__" function'
        )

    def __len__(self) -> None:
        raise NotImplementedError(
            f'{type(self).__name__} is missing the implementation of "__len__" function'
        )
