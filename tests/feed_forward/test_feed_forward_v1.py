import torch
import torch.nn as nn

from llmtutorial.feed_forward.feed_forward_v1 import FeedForwardV1


# pytest -sv tests/feed_forward/test_feed_forward_v1.py
class TestFeedForwardV1:

    _seed_num = 123
    _embedding_dim = 768
    _activ_func = nn.ReLU()
    _batch_size = 2
    _cxt_len = 3

    def test_emb_dim_size(self):
        torch.manual_seed(self._seed_num)
        ffn = FeedForwardV1(self._embedding_dim, self._activ_func)
        x = torch.rand(self._batch_size, self._cxt_len, self._embedding_dim)
        out = ffn(x)
        assert x.shape == out.shape
