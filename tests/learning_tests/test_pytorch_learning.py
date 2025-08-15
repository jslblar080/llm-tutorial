import torch
import pytest


class TestPytorchLearning:
    skip_remaining = False

    @pytest.fixture(autouse=True)
    def check_skip_remaining(self):
        if TestPytorchLearning.skip_remaining:
            pytest.skip("Skipping remaining tests because CUDA is not available")

    def test_torch_version(self):
        minimum_version = "2.3.0"
        assert (
            torch.__version__ >= minimum_version
        ), f"PyTorch version mismatch: Expected at least {minimum_version}, got {torch.__version__}"

    def test_cuda_availability(self):
        if not torch.cuda.is_available():
            TestPytorchLearning.skip_remaining = True
            pytest.skip("CUDA is not available, skipping remaining tests")

    # TODO: Update multiple learning tests with Pytorch

    def test_three(self):
        x = "this"
        assert "h" in x

    def test_four(self):
        x = "hello"
        assert hasattr(x, "check")
