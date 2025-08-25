import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader


class TestPytorchLearning:

    class NeuralNetwork(nn.Module):

        _layers: nn.Sequential

        def __init__(self, num_inputs: int, num_outputs: int) -> None:
            super().__init__()

            self._layers = nn.Sequential(
                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),
                # 2nd hidden layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),
                # output layer
                torch.nn.Linear(20, num_outputs),
            )

        def forward(self, x):
            """
            Forward method.

            This is the entry point for computation when calling the model instance.

            PyTorch internal flow:
            1. `model.__call__(...)` → triggers the call.
            2. `_wrapped_call_impl(...)` → chooses compiled call or normal.
            3. `_call_impl(...)` → runs forward safely, handles hooks, and tracing.
            4. `self.forward(...)` → executes the user-defined computation.

            Note:
            Do not call `forward()` directly; use `model(...)` or a wrapper method.
            """
            logits = self._layers(x)
            return logits

    class ToyDataset(Dataset):

        _features: Tensor
        _labels: Tensor

        def __init__(self, X: Tensor, y: Tensor) -> None:
            self._features = X
            self._labels = y

        def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
            one_x = self._features[index]
            one_y = self._labels[index]
            return one_x, one_y

        def __len__(self) -> int:
            return self._labels.shape[0]

    _skip_remaining = False

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

    @staticmethod
    def compute_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
        model.eval()
        correct = 0
        total_examples = 0
        device = next(model.parameters()).device
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)  # forward
                predictions = torch.argmax(logits, dim=1)
                is_correct = (predictions == labels).sum().item()
                correct += is_correct
                total_examples += labels.size(0)
        return correct / total_examples

    @pytest.fixture(autouse=True)
    def check_skip_remaining(self):
        if type(self)._skip_remaining:
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

    def test_module_subclass(self):
        torch.manual_seed(123)
        model = self.NeuralNetwork(50, 3)
        X = torch.rand((1, 50))
        with torch.no_grad():  # prediction without training
            out = torch.softmax(model(X), dim=1)
        torch.testing.assert_close(torch.sum(out), torch.ones(()))

    def test_dataset_subclass_dataloader(self):
        X_train = torch.tensor(
            [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
        )
        y_train = torch.tensor([0, 0, 0, 1, 1])
        train_ds = self.ToyDataset(X_train, y_train)
        for i in range(train_ds.__len__()):
            torch.testing.assert_close(
                train_ds.__getitem__(i), (X_train[i], y_train[i])
            )
        X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
        y_test = torch.tensor([0, 1])
        test_ds = self.ToyDataset(X_test, y_test)
        for i in range(test_ds.__len__()):
            torch.testing.assert_close(test_ds.__getitem__(i), (X_test[i], y_test[i]))

        torch.manual_seed(123)
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            drop_last=True,  # batch
        )
        test_loader = DataLoader(
            dataset=test_ds, batch_size=2, shuffle=False, num_workers=0
        )
        model = self.NeuralNetwork(
            2, 2  # 2 input features / 2 class labels (0, 1) to predict
        )
        # SGD: stochastic gradient descent / lr(learning rate): tuning with loss convergence
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        num_epochs = 3  # epoch: tuning with loss convergence
        print()
        for epoch in range(num_epochs):
            model.train()  # necessary for dropout or batch normalization layers
            for batch_idx, (features, labels) in enumerate(train_loader):
                logits = model(features)  # forward
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()  # prevent gradient accumulation from previous batch
                loss.backward()  # compute gradients of loss with computation graph
                optimizer.step()  # use gradients for model parameters update to minimize loss
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                    f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                    f" | Train/Val Loss: {loss:.2f}"
                )
            model.eval()  # necessary for dropout or batch normalization layers
        torch.testing.assert_close(
            type(self).compute_accuracy(model, train_loader), 1.0
        )
        torch.testing.assert_close(type(self).compute_accuracy(model, test_loader), 1.0)

    def test_cuda_availability(self):
        if not torch.cuda.is_available():
            type(self)._skip_remaining = True
            pytest.skip("CUDA is not available, skipping remaining tests")

    def test_fail(self):
        x = "hello"
        assert hasattr(x, "check")
