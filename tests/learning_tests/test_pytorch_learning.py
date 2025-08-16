import torch
import pytest


class TestPytorchLearning:
    skip_remaining = False

    @staticmethod
    def to_onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
        y_onehot = torch.zeros(y.size(0), num_classes)
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
        return y_onehot

    @staticmethod
    def softmax(z: torch.Tensor) -> torch.Tensor:
        # subtract max for numerical stability
        z_max, _ = torch.max(z, dim=1, keepdim=True)
        exp_z = torch.exp(z - z_max)
        return exp_z / torch.sum(exp_z, dim=1, keepdim=True)

    @staticmethod
    def to_classlabel(z: torch.Tensor) -> torch.Tensor:
        return torch.argmax(z, dim=1)

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
        expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        result = type(self).to_onehot(torch.tensor([0, 1, 2, 2]), 3)
        assert torch.equal(result, expected), f"Expected:\n{expected}\nGot:\n{result}"

    def test_softmax(self):
        predicted = type(self).to_classlabel(
            type(self).softmax(
                torch.tensor(
                    [
                        [-0.3, -0.5, -0.5],
                        [-0.4, -0.1, -0.5],
                        [-0.3, -0.94, -0.5],
                        [-0.99, -0.88, -0.5],
                    ]
                )
            )
        )
        true_labels = type(self).to_classlabel(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            )
        )
        accuracy = (predicted == true_labels).sum().item() / predicted.size(0)
        assert (
            accuracy >= 0.75
        ), f"Accuracy too low: {accuracy*100:.1f}% - predicted: {predicted}, true: {true_labels}"

    # TODO: Update cross entropy test with Pytorch

    def test_cuda_availability(self):
        if not torch.cuda.is_available():
            type(self).skip_remaining = True
            pytest.skip("CUDA is not available, skipping remaining tests")

    def test_fail(self):
        x = "hello"
        assert hasattr(x, "check")
