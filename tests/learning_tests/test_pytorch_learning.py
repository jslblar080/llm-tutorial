import torch
import pytest


class TestPytorchLearning:
    skip_remaining = False

    @staticmethod
    def to_onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
        y_onehot = torch.zeros(y.size(0), num_classes)
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
        return y_onehot

    @pytest.fixture(autouse=True)
    def check_skip_remaining(self):
        if type(self).skip_remaining:
            pytest.skip("Skipping remaining tests because CUDA is not available")

    def test_torch_version(self):
        minimum_version = "2.3.0"
        assert (
            torch.__version__ >= minimum_version
        ), f"PyTorch version mismatch: Expected at least {minimum_version}, got {torch.__version__}"

    def test_onehot_encoding(self):
        y = torch.tensor([0, 1, 2, 2])
        expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        result = type(self).to_onehot(y, 3)
        assert torch.equal(result, expected), f"Expected:\n{expected}\nGot:\n{result}"

    # TODO: Update softmax test with Pytorch

    def test_cuda_availability(self):
        if not torch.cuda.is_available():
            type(self).skip_remaining = True
            pytest.skip("CUDA is not available, skipping remaining tests")

    def test_fail(self):
        x = "hello"
        assert hasattr(x, "check")
