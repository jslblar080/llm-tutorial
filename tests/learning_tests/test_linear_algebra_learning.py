import numpy as np
from typing import Callable, TypeAlias


# pytest -sv tests/learning_tests/test_linear_algebra_learning.py
class TestLinearAlgebraLearning:

    class LinearityTester:

        Vector: TypeAlias = np.ndarray
        _system: Callable[[Vector], Vector]
        _formula: str

        def __init__(self, system: Callable[[Vector], Vector], formula: str) -> None:
            self._system = system
            self._formula = formula

        def check_superposition(self) -> bool:
            x1 = np.random.randn(30)
            x2 = np.random.randn(30)
            lhs = self._system(x1 + x2)
            rhs = self._system(x1) + self._system(x2)
            return np.allclose(lhs, rhs)

        def check_homogeneity(self) -> bool:
            alpha = np.random.randn()
            x = np.random.randn(30)
            lhs = self._system(alpha * x)
            rhs = alpha * self._system(x)
            return np.allclose(lhs, rhs)

        def check_linearity(self) -> bool:
            print(f"\n{self._formula}")
            print("Superposition:", self.check_superposition())
            print("Homogeneity:", self.check_homogeneity())
            is_linear = self.check_superposition() and self.check_homogeneity()
            print("Linearity:", is_linear)
            return is_linear

    def test_linearity(self):
        linear_system_1 = lambda x: 2 * x, "T(x) = 2 * x"
        assert self.LinearityTester(*linear_system_1).check_linearity()
        linear_system_2 = lambda f: np.gradient(f), r"T(f) = \frac{df}{dx}"
        assert self.LinearityTester(*linear_system_2).check_linearity()
        linear_system_3 = lambda f: np.cumsum(f), r"T(f) = \int f \, dx"
        assert self.LinearityTester(*linear_system_3).check_linearity()
        non_linear_system_1 = lambda x: 2 * x**2, "T(x) = 2 * x**2"
        assert not self.LinearityTester(*non_linear_system_1).check_linearity()
        non_linear_system_2 = lambda x: 2 * x + 3, "T(x) = 2 * x + 3"
        assert not self.LinearityTester(*non_linear_system_2).check_linearity()
