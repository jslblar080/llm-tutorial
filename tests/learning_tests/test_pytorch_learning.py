import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad


class TestPytorchLearning:
    skip_remaining = False

    @staticmethod
    def to_onehot(y: Tensor, num_classes: int) -> Tensor:
        y_onehot = torch.zeros(y.size(0), num_classes)
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
        return y_onehot

    @staticmethod
    def softmax(z: Tensor) -> Tensor:
        # subtract max for numerical stability
        z_max, _ = torch.max(z, dim=1, keepdim=True)
        exp_z = torch.exp(z - z_max)
        return exp_z / torch.sum(exp_z, dim=1, keepdim=True)

    @staticmethod
    def to_classlabel(z: Tensor) -> Tensor:
        return torch.argmax(z, dim=1)

    @staticmethod
    def cross_entropy(softmax: Tensor, y_target: Tensor) -> Tensor:
        return -torch.sum(torch.log(softmax) * (y_target), dim=1)

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

    def test_cross_entropy(self):
        Z = torch.tensor(
            [
                [-0.3, -0.5, -0.5],
                [-0.4, -0.1, -0.5],
                [-0.3, -0.94, -0.5],
                [-0.99, -0.88, -0.5],
            ]
        )
        y = torch.tensor([0, 1, 2, 2])
        xent = type(self).cross_entropy(
            type(self).softmax(Z),
            type(self).to_onehot(y, 3),
        )
        assert torch.all(xent >= 0), "Cross entropy must be non-negative"
        assert torch.allclose(
            F.nll_loss(torch.log(type(self).softmax(Z)), y, reduction="none"),
            F.cross_entropy(Z, y, reduction="none"),
        ), "PyTorch nll_loss(log(softmax)) and cross_entropy results differ"
        assert torch.allclose(
            xent, F.cross_entropy(Z, y, reduction="none")
        ), "Custom cross_entropy does not match PyTorch cross_entropy"
        assert torch.allclose(
            torch.mean(xent), F.cross_entropy(Z, y)
        ), "Mean of custom cross_entropy does not match PyTorch's reduced cross_entropy"

    def test_backward_grad_first(self):
        y = torch.tensor([1.0])  # target
        x1 = torch.tensor([1.1])  # input
        w1 = torch.tensor([2.2], requires_grad=True)  # weight
        b = torch.tensor([0.0], requires_grad=True)  # bias
        z = w1 * x1 + b  # net input
        a = torch.sigmoid(z)  # output (activation)
        loss = F.binary_cross_entropy(a, y)
        (grad_L_w1,) = grad(loss, w1, retain_graph=True)
        (grad_L_b,) = grad(loss, b, retain_graph=True)
        loss.backward()
        assert w1.grad is not None
        assert b.grad is not None
        assert torch.equal(grad_L_w1, w1.grad)
        assert torch.equal(grad_L_b, b.grad)

    def test_backward_backward_first(self):
        y = torch.tensor([1.0])  # target
        x1 = torch.tensor([1.1])  # input
        w1 = torch.tensor([2.2], requires_grad=True)  # weight
        b = torch.tensor([0.0], requires_grad=True)  # bias
        z = w1 * x1 + b  # net input
        a = torch.sigmoid(z)  # output (activation)
        loss = F.binary_cross_entropy(a, y)
        loss.backward(retain_graph=True)
        (grad_L_w1,) = grad(loss, w1, retain_graph=True)
        (grad_L_b,) = grad(loss, b)
        assert w1.grad is not None
        assert b.grad is not None
        assert torch.equal(grad_L_w1, w1.grad)
        assert torch.equal(grad_L_b, b.grad)

    def test_cuda_availability(self):
        if not torch.cuda.is_available():
            type(self).skip_remaining = True
            pytest.skip("CUDA is not available, skipping remaining tests")

    def test_fail(self):
        x = "hello"
        assert hasattr(x, "check")
