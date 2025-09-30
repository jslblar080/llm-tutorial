import typer

from .config import Config
from .gpt_model.gpt_model_config import GPTModelConfig
from .gpt_model.gpt_model_v1_config import GPTModelV1Config
from .util.singleton_meta import SingletonMeta


class CLI(metaclass=SingletonMeta):

    app: typer.Typer
    succeeded: bool

    def __init__(self):
        self.app = typer.Typer()
        self.succeeded = False
        self.app.command()(self.config)
        self.app.command()(self.gptmodelconfig)
        self.app.command()(self.gptmodelv1config)

    def config(
        self,
        textfile="the-verdict.txt",
        trainratio=0.9,
        seednum=123,
        batchsize=2,
        numworkers=0,
        cxtlen=256,
        encoding="gpt2",
        dataset="GPTDatasetV1",
        gptmodel="GPTModelV1",
    ):
        """Set parameters for config.py"""
        self.succeeded = False
        typer.echo(
            f"""
Text path: {textfile}
Train ratio: {trainratio}
Seed number: {seednum}
Batch size: {batchsize}
Number of workers: {numworkers}
Context length: {cxtlen}
Encoding: {encoding}
Dataset: {dataset}
GPT Model: {gptmodel}"""
        )
        config = Config()
        path_except_last = config.texts[:-1]
        config.texts = (*path_except_last, textfile)
        config.train_ratio = float(trainratio)
        config.seed_num = int(seednum)
        config.batch_size = int(batchsize)
        config.num_workers = int(numworkers)
        config.context_length = int(cxtlen)
        config.encoding = encoding
        config.dataset_flags.set(dataset)
        GPTModelConfig().initialize()
        GPTModelV1Config().initialize()
        config.gpt_model_flags.set(gptmodel)
        config.gpt_model = config.seed_num
        self.succeeded = True

    def gptmodelconfig(self, attention="MultiHeadAttention"):
        """Set parameters for gpt_model_config.py"""
        self.succeeded = False
        # TODO
        typer.echo(
            f"""
Attention: {attention}"""
        )
        config = Config()
        gptmodelconfig = GPTModelConfig()
        # TODO
        gptmodelconfig.attention_flags.set(attention)
        gptmodelconfig.attention = config.seed_num
        GPTModelV1Config().initialize()
        config.gpt_model = config.seed_num
        self.succeeded = True

    def gptmodelv1config(self):
        """Set parameters for gpt_model_v1_config.py"""
        self.succeeded = False
        # TODO
        config = Config()
        gptmodelconfig = GPTModelConfig()
        gptmodelv1config = GPTModelV1Config()
        # TODO
        config.gpt_model = config.seed_num
        # self.succeeded = True

    # TODO: sequential multi command
