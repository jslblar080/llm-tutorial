import matplotlib.pyplot as plt
import numpy as np
import os
import sympy
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

    @staticmethod
    def tensor_from(
        shape: tuple[int, ...], formula: Callable, dtype=float
    ) -> np.ndarray:
        """
        list of all 1’s, but replace the axis entry with size
        """
        axes = [
            np.arange(size, dtype=dtype).reshape(
                *((1,) * axis + (size,) + (1,) * (len(shape) - axis - 1))
            )
            for axis, size in enumerate(shape)
        ]
        return formula(*axes)

    @staticmethod
    def calc_mai(mat: np.ndarray) -> float:
        assert not np.all(mat == 0)
        anti_mat = (mat - mat.T) / 2
        """
        matrix asymmetry index

        symmetric matrix -> 0 
        skew-symmetric matrix -> 1
        """
        mai = float(np.linalg.norm(anti_mat) / np.linalg.norm(mat))
        return mai

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
        """
        sum of vector hadamard (element-wise) multiplication
        """
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

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_dot_product_geometry
    def test_dot_product_geometry(self):
        v1 = np.array([2, 4, -3])
        v2 = np.array([0, -3, -3])
        # arccos: inverse of cosine
        ang = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        assert isinstance(ax, Axes3D), f"Expected a 3D Axes, got {type(ax)} instead."
        ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]], color="b", linewidth=2)
        ax.plot([0, v2[0]], [0, v2[1]], [0, v2[2]], color="r", linewidth=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"Angle between vectors: {ang} rad.")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        save_path = os.path.join(script_dir, "outputs", "3d_vector_dot_product.png")
        plt.savefig(save_path)
        plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_cauchy_schwarz_inequality
    def test_cauchy_schwarz_inequality(self):
        n = np.random.randint(100)
        v1 = np.random.randn(n)
        v2 = np.random.randn(n)
        """
        inequality
        """
        assert np.abs(np.dot(v1, v2)) < np.linalg.norm(v1) * np.linalg.norm(v2)
        v1_scaled = np.random.randn(1) * v1
        """
        equality
        """
        assert np.isclose(
            np.abs(np.dot(v1, v1_scaled)),
            np.linalg.norm(v1) * np.linalg.norm(v1_scaled),
        )

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_outer_product
    def test_outer_product(self):
        d1 = 3
        d2 = 4
        v1 = np.random.randn(d1)
        v2 = np.random.randn(d2)
        op1 = np.outer(v1, v2)
        op2 = np.zeros((d1, d2))
        for i in range(d1):
            for j in range(d2):
                op2[i, j] = v1[i] * v2[j]
        assert np.allclose(op1, op2)

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_vector_cross_product
    def test_vector_cross_product(self):
        v1 = np.array([-3, 2, 5])
        v2 = np.array([4, -3, 0])
        v3a = np.cross(v1, v2)
        v3b = np.array(
            [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]
        )
        assert np.allclose(v3a, v3b)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        assert isinstance(ax, Axes3D), f"Expected a 3D Axes, got {type(ax)} instead."
        ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]], color="k", linewidth=2)
        ax.plot([0, v2[0]], [0, v2[1]], [0, v2[2]], color="k", linewidth=2)
        ax.plot([0, v3a[0]], [0, v3a[1]], [0, v3a[2]], color="r", linewidth=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        save_path = os.path.join(script_dir, "outputs", "3d_vector_cross_product.png")
        plt.savefig(save_path)
        plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_hermitian_transpose
    def test_hermitian_transpose(self):
        z = np.array([3, 4j, 5 + 2j, 2 - 5j])
        z_H_z = np.transpose(np.conjugate(z)) * z
        assert np.allclose(z_H_z, np.array([9, 16, 29, 29]))
        print(z_H_z)

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_unit_vector
    def test_unit_vector(self):
        origin = np.array([0, 0])
        v = np.array([-3, 6])
        mu = 1 / np.linalg.norm(v)
        unit_v = v * mu
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
            *unit_v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            label="unit_v",
        )
        plt.legend()
        plt.axis("square")
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.grid()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        save_path = os.path.join(script_dir, "outputs", "unit_vector.png")
        plt.savefig(save_path)
        plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_matrix_multiplication_via_layers
    def test_matrix_multiplication_via_layers(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 6)
        spectral = np.zeros((A.shape[0], B.shape[1]))
        inner_dim = A.shape[1]
        assert inner_dim == B.shape[0]
        for i in range(A.shape[1]):
            spectral += np.outer(A[:, i], B[i, :])
        print(spectral)
        assert np.allclose(spectral, A @ B)

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_standard_hadamard_matrix_multiplication_diagonal_matrix
    def test_standard_hadamard_matrix_multiplication_diagonal_matrix(self):
        n = 100
        F = np.random.randn(n, n)
        D = np.diag(np.random.randn(n))
        assert not np.allclose(F @ F, F * F)
        assert np.allclose(D @ D, D * D)

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_discrete_fourier_transform
    def test_discrete_fourier_transform(self):
        """
        DFT (Discrete Fourier Transform): determine frequency content of discrete-time signal
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        N = 64
        t = np.arange(N)
        T1, T2 = 8, 16
        signal = np.sin(2 * np.pi * t / T1) + 0.5 * np.sin(2 * np.pi * t / T2)
        np.random.seed(123)
        noise = 0.3 * np.random.randn(N)
        """
        x: discrete-time signal vector with noise
           (all elements are real numbers in this case)
        """
        x = signal + noise
        plt.plot(t, x, label="discrete-time signal with noise")
        plt.legend()
        save_path = os.path.join(
            script_dir, "outputs", "DFT_discrete_time_signal_with_noise.png"
        )
        plt.savefig(save_path)
        plt.close()
        """
        F: DFT matrix (N x N)
        """
        F = self.tensor_from(
            (N, N), lambda k, n: np.exp(1) ** (-2j * np.pi * k * n / N)
        )
        X = F @ x
        assert np.allclose(X, np.fft.fft(x))
        plt.stem(np.abs(X))
        plt.title("Magnitude Spectrum")
        """
        x consists only of real numbers
        -> magnitude spectrum (|X|) is conjugate symmetric about k = N/2
        """
        save_path = os.path.join(script_dir, "outputs", "DFT_magnitude_spectrum.png")
        plt.savefig(save_path)
        plt.close()
        """
        filter out all DFT indices except those (k2, k1)
        corresponding to first and second largest magnitudes in spectrum
        """
        X_filtered = np.zeros_like(X)
        k1, k2 = int(N / T1), int(N / T2)
        X_filtered[k2] = X[k2]
        X_filtered[k1] = X[k1]
        X_filtered[N - k1] = X[N - k1]
        X_filtered[N - k2] = X[N - k2]
        """
        IDFT (Inverse Discrete Fourier Transform)
        """
        x_filtered = np.real(np.conj(F.T) @ X_filtered / N)
        assert np.allclose(
            x_filtered, np.fft.ifft(X_filtered)
        )  # np.fft.ifft(X_filtered).real
        plt.plot(t, x, label="original signal with noise")
        plt.plot(t, x_filtered, label="filtered signal")
        plt.legend()
        save_path = os.path.join(script_dir, "outputs", "DFT_filtered_signal.png")
        plt.savefig(save_path)
        plt.close()

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_frobenius_dot_product
    def test_frobenius_dot_product(self):
        m, n = 9, 4
        A = np.random.randn(m, n)
        B = np.random.randn(m, n)
        print()
        """
        total sum of hadamard matrix multiplication
        """
        frob_dp1 = np.sum(A * B)
        print(frob_dp1)
        """
        vectorize -> dot product
        """
        a = np.reshape(A, (m * n))
        b = np.reshape(B, (m * n))
        frob_dp2 = np.dot(a, b)
        print(frob_dp2)
        assert np.isclose(frob_dp1, frob_dp2)
        """
        transpose-trace
        """
        frob_dp3 = np.trace(A.T @ B)
        print(frob_dp3)
        assert np.isclose(frob_dp3, np.trace(B.T @ A))
        assert np.isclose(frob_dp1, frob_dp3)

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_matrix_asymmetry_index
    def test_matrix_asymmetry_index(self):
        n = 5
        A = np.random.randn(n, n)
        assert 0 <= self.calc_mai(A) <= 1
        sym_A = (A + A.T) / 2
        assert self.calc_mai(sym_A) == 0
        skew_sym_A = (A - A.T) / 2
        assert self.calc_mai(skew_sym_A) == 1
        p = np.random.rand()
        B = (1 - p) * (A + A.T) + p * (A - A.T)
        assert 0 <= self.calc_mai(B) <= 1

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_rank_computation
    def test_rank_computation(self):
        m, n = 4, 6
        A = np.random.randn(m, n)
        rank_A = np.linalg.matrix_rank(A)
        print(f"\nrank(A): {rank_A}")
        assert rank_A == min([m, n])
        B = A
        B[:, -1] = B[:, -2]
        rank_B = np.linalg.matrix_rank(B)
        print(f"rank(B): {rank_B}")
        assert rank_B == min([m, n - 1])
        C = A
        C[-1, :] = C[-2, :]
        rank_C = np.linalg.matrix_rank(C)
        print(f"rank(C): {rank_C}")
        assert rank_C == min([m - 1, n])
        F = np.round(10 * np.random.randn(m, m))
        F[:, -1] = F[:, -2]
        """
        make square matrix full-rank by shifting (add small noise)
        F~ = F + λI
        """
        noise_lambda = 0.000001
        F_tilde = F + noise_lambda * np.eye(m, m)
        print(f"reduced-rank without shifting: {np.linalg.matrix_rank(F)}")
        print(f"full-rank with shifting: {np.linalg.matrix_rank(F_tilde)}")
        assert np.linalg.matrix_rank(F) == m - 1
        assert np.linalg.matrix_rank(F_tilde) == m
        """
        mxn matrix with reduced-rank r via multiplication
        rank(A @ B) <= min(rank(A), rank(B))
        C = A @ B
        ci = ai @ B
        cj = A @ bj
        """
        m, n = np.random.randint(51, 100, size=2)
        r = np.random.randint(1, 50)
        A = np.random.randn(m, r)
        B = np.random.randn(r, n)
        C = A @ B
        assert np.linalg.matrix_rank(C) == r
        """
        rank(C @ C.T) == rank(C) == rank(C.T) == rank(C.T @ C)
        """
        assert np.linalg.matrix_rank(C @ C.T) == np.linalg.matrix_rank(C)
        assert np.linalg.matrix_rank(C) == np.linalg.matrix_rank(C.T)
        assert np.linalg.matrix_rank(C.T) == np.linalg.matrix_rank(C.T @ C)
        """
        determine whether vector(v) is in span of set(S or T) via augmentation
        """
        v = np.array([1, 2, 3, 4])
        print()
        for set_name, set in (
            ("S", np.array([[4, 3, 6, 2], [0, 4, 0, 1]])),
            ("T", np.array([[1, 2, 2, 2], [0, 0, 1, 2]])),
        ):
            set_stacked = np.vstack([set, v])
            if np.linalg.matrix_rank(set_stacked) == 3:
                print(f"vector v {v}\nis not in span of set {set_name}\n{set}")
            elif np.linalg.matrix_rank(set_stacked) < 3:
                print(f"vector v {v}\nis in span of set {set_name}\n{set}")
            else:
                return

    # pytest -sv tests/learning_tests/test_linear_algebra_learning.py::TestLinearAlgebraLearning::test_reduced_row_echelon_form
    def test_reduced_row_echelon_form(self):
        m, n = 6, 4
        assert m >= n
        A = sympy.Matrix(np.random.randn(m, m))
        B = sympy.Matrix(np.random.randn(m, n))
        C = sympy.Matrix(np.random.randn(n, m))
        D = sympy.Matrix(np.random.randn(n, n))
        D[:, 0] = D[:, 1]
        A_rref = np.array(A.rref()[0])
        B_rref = np.array(B.rref()[0])
        C_rref = np.array(C.rref()[0])
        D_rref = np.array(D.rref()[0])
        print(f"\n{A_rref}")
        assert sum(np.any(A_rref != 0, axis=1)) == np.linalg.matrix_rank(
            sympy.matrix2numpy(A, dtype=float)
        )
        print(f"\n{B_rref}")
        assert sum(np.any(B_rref != 0, axis=1)) == np.linalg.matrix_rank(
            sympy.matrix2numpy(B, dtype=float)
        )
        print(f"\n{C_rref}")
        assert sum(np.any(C_rref != 0, axis=1)) == np.linalg.matrix_rank(
            sympy.matrix2numpy(C, dtype=float)
        )
        print(f"\n{D_rref}")
        assert sum(np.any(D_rref != 0, axis=1)) == np.linalg.matrix_rank(
            sympy.matrix2numpy(D, dtype=float)
        )
