from util.singleton_meta import SingletonMeta


class Config(metaclass=SingletonMeta):

    def __init__(self) -> None:
        self.texts = ["""Hello, world.""", """This, is a test."""]
        self.encoding = "o200k_base"
        self.embedding_dim = 3
