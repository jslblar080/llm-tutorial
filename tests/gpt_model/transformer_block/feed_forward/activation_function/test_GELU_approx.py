import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

from llmtutorial.gpt_model.transformer_block.feed_forward.activation_function.GELU_approx import GELUApprox


# pytest -sv tests/gpt_model/transformer_block/feed_forward/activation_function/test_GELU_approx.py
class TestGELUApprox:

    def test_gelu_approx_vs_relu(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "gelu_approx_vs_relu.png")
        gelu_approx, relu = GELUApprox(), nn.ReLU()
        x = torch.linspace(-3, 3, 100)
        y_gelu, y_relu = gelu_approx(x), relu(x)
        plt.figure(figsize=(10, 4))
        for i, (y, label) in enumerate(
            zip([y_gelu, y_relu], ["GELUApprox", "nn.ReLU"]), 1
        ):
            plt.subplot(1, 2, i)
            plt.plot(x, y)
            plt.title(f"{label} activation function")
            plt.xlabel("x")
            plt.ylabel(f"{label}(x)")
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print("\nImage saved to:", save_path)
