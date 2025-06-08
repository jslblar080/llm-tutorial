from util.singleton_meta import SingletonMeta


class Config(metaclass=SingletonMeta):

    def __init__(self) -> None:
        self.text = "Hello, world. This, is a test."
        self.embedding_dim = 16
