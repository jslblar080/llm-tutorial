import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
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

    def test_singular_value_decomposition(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        file_path = os.path.join(script_dir, "assets", "cafe_rooted.jpeg")
        pic = Image.open(file_path).convert("L")  # "L" = 8-bit grayscale
        pic = np.array(pic)
        save_path_1 = os.path.join(script_dir, "outputs", "cafe_rooted_grayscale.png")
        plt.imsave(save_path_1, pic, cmap="gray")
        plt.close()
        """
        SVD (singular value decomposition)
        """
        U, S, V = np.linalg.svd(pic)
        plt.plot(S, "s-")
        plt.xlim([0, 50])
        plt.xlabel("Component number")
        plt.ylabel("Singular value")
        save_path_2 = os.path.join(script_dir, "outputs", "cafe_rooted_scree_plot.png")
        plt.savefig(save_path_2)
        plt.close()
        """
        Low-rank approximation
        """
        for start_comp, num_comps in (
            (0, 10),
            (0, 20),
            (0, 30),
            (0, 40),
            (0, 50),
            (20, 130),
        ):
            comps = np.arange(start_comp, num_comps)
            recon_pic = U[:, comps] @ np.diag(S[comps]) @ V[comps, :]
            recon_pic_title = (
                "cafe_rooted_comp_"
                + str(start_comp)
                + "-"
                + str(start_comp + num_comps - 1)
                + ".png"
            )
            save_path_3 = os.path.join(script_dir, "outputs", recon_pic_title)
            plt.imsave(save_path_3, recon_pic, cmap="gray")
            plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_3d_vector_transpose_plot
    def test_3d_vector_transpose_plot(self):
        """
        3-dimensional vector
        """
        v3_tensor = torch.tensor([4, -3, 2])
        v3t_np = np.transpose(v3_tensor.numpy())
        v3tT_np = v3t_np.T
        assert np.array_equal(v3tT_np, v3_tensor.numpy())
        assert torch.equal(torch.tensor(v3tT_np), v3_tensor)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        assert isinstance(ax, Axes3D), f"Expected a 3D Axes, got {type(ax)} instead."
        ax.plot(
            [0, v3tT_np[0]], [0, v3tT_np[1]], [0, v3tT_np[2]], color="r", linewidth=2
        )
        ax.plot([-4, 4], [0, 0], [0, 0], "k--")
        ax.plot([0, 0], [-4, 4], [0, 0], "k--")
        ax.plot([0, 0], [0, 0], [-4, 4], "k--")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        save_path = os.path.join(script_dir, "outputs", "3d_vector_example.png")
        plt.savefig(save_path)
        plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_vector_addition
    def test_vector_addition(self):
        origin = np.array([0, 0])
        v1 = np.array([3, -1])
        v2 = np.array([2, 4])
        v1_plus_v2 = v1 + v2  # list just concatenates with +
        plt.quiver(
            *origin,
            *v1,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="b",
            label="v1",
        )
        plt.quiver(
            *v1,
            *v2,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            label="v2",
        )
        plt.quiver(
            *origin,
            *v1_plus_v2,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="k",
            label="v1 + v2",
        )
        plt.legend()
        plt.axis("equal")
        plt.xlim(-1, 7)
        plt.ylim(-3, 7)
        plt.grid()
        plt.title("Vector Addition: v1 + v2")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        save_path = os.path.join(script_dir, "outputs", "2d_vector_addition.png")
        plt.savefig(save_path)
        plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_vector_scalar_multiplication
    def test_vector_scalar_multiplication(self):
        origin = np.array([0, 0])
        v = np.array([3, -1])
        scalar = -0.3
        scalar_times_v = scalar * v
        plt.quiver(
            *origin,
            *v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="b",
            label="v",
        )
        plt.quiver(
            *origin,
            *scalar_times_v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            label="scalar * v",
        )
        plt.legend()
        plt.axis("equal")
        plt.xlim(-2, 4)
        plt.ylim(-2, 1)
        plt.grid()
        plt.title("Vector Scalar Multiplication: scalar * v")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        save_path = os.path.join(
            script_dir, "outputs", "2d_vector_scalar_multiplication.png"
        )
        plt.savefig(save_path)
        plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_dot_product_property
    def test_dot_product_property(self):
        n = 10
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)
        dp1 = np.dot(v1, v2)
        dp2 = np.matmul(v1, v2)
        dp3 = sum(np.multiply(v1, v2))
        dp4 = 0
        for i in range(0, len(v1)):
            dp4 = dp4 + v1[i] * v2[i]
        assert np.isclose(dp1, dp2, dp3, dp4)
        """
        dot product is distributive
        """
        v3 = np.random.rand(n)
        res1 = np.dot(v1, (v2 + v3))
        res2 = np.dot(v1, v2) + np.dot(v1, v3)
        assert np.isclose(res1, res2)
        """
        dot product is not associative
        unless one vector is zeros vector or all vectors are same
        """
        res1 = np.dot(v1, np.dot(v2, v3))
        res2 = np.dot(np.dot(v1, v2), v3)
        assert not np.allclose(res1, res2)
        """
        dot product is commutative
        """
        res1 = np.dot(v1, v2)
        res2 = np.dot(v2, v1)
        assert np.isclose(res1, res2)

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_vector_length_norm
    def test_vector_length_norm(self):
        n = 10
        v = np.random.rand(n)
        vl1 = np.sqrt(np.dot(v, v))
        vl2 = np.linalg.norm(v)
        assert np.isclose(vl1, vl2)
