import pytest
import torch

from torch import Tensor
from llmtutorial.gpt_model.layer_norm.layer_norm_v1 import LayerNormV1


# pytest -sv tests/gpt_model/layer_norm/test_layer_norm_v1.py
class TestLayerNormV1:

    _seed_num = 123
    _batch_size = 2
    _embedding_dim = 5
    _batch_example: Tensor
    _ln__v1: LayerNormV1
    _out_ln_v1: Tensor

    @pytest.fixture(autouse=True)
    def keep_manual_seed(self):
        torch.manual_seed(self._seed_num)
        self._batch_example = torch.randn(self._batch_size, self._embedding_dim)
        self._ln__v1 = LayerNormV1(self._embedding_dim)
        self._out_ln_v1 = self._ln__v1(self._batch_example)
        torch.set_printoptions(sci_mode=False)

    def test_mean(self):
        mean = self._out_ln_v1.mean(dim=-1, keepdim=True)
        print("\nMean:\n", mean)
        torch.allclose(mean, torch.zeros(self._batch_size))

    def test_var(self):
        var = self._out_ln_v1.var(dim=-1, unbiased=False, keepdim=True)
        print("\nVariance:\n", var)
        torch.allclose(var, torch.ones(self._batch_size))
