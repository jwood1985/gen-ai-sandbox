"""
Tests for the GPT-style feedforward block.

Verifies:
  1. Output shapes match input shapes.
  2. Analytical gradients match numerical (finite-difference) gradients.
  3. Residual connection is present (output != pure FFN output).
  4. Inference mode (training=False) is deterministic.
"""

import numpy as np
import pytest

from feedforward import FeedForwardBlock, gelu, gelu_backward, LayerNorm, Linear


# ── helpers ──────────────────────────────────────────────────────────

def numerical_gradient(f, x, eps=1e-5):
    """Compute the numerical gradient of scalar function f w.r.t. array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f()
        x[idx] = old - eps
        fm = f()
        grad[idx] = (fp - fm) / (2.0 * eps)
        x[idx] = old
        it.iternext()
    return grad


# ── shape tests ──────────────────────────────────────────────────────

class TestShapes:
    def test_forward_shape(self):
        np.random.seed(0)
        ffn = FeedForwardBlock(d_model=32)
        x = np.random.randn(2, 4, 32)
        out = ffn.forward(x, training=False)
        assert out.shape == x.shape

    def test_backward_shape(self):
        np.random.seed(0)
        ffn = FeedForwardBlock(d_model=32)
        x = np.random.randn(2, 4, 32)
        ffn.forward(x, training=False)
        dx = ffn.backward(np.ones_like(x))
        assert dx.shape == x.shape

    def test_single_sample(self):
        np.random.seed(0)
        ffn = FeedForwardBlock(d_model=16)
        x = np.random.randn(1, 1, 16)
        out = ffn.forward(x, training=False)
        assert out.shape == (1, 1, 16)


# ── GELU tests ───────────────────────────────────────────────────────

class TestGELU:
    def test_gelu_zero(self):
        assert np.isclose(gelu(np.array([0.0]))[0], 0.0, atol=1e-7)

    def test_gelu_positive(self):
        assert gelu(np.array([2.0]))[0] > 0.0

    def test_gelu_gradient_numerical(self):
        np.random.seed(1)
        x = np.random.randn(5)
        analytical = gelu_backward(x)
        eps = 1e-5
        numerical = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)


# ── Linear layer gradient check ─────────────────────────────────────

class TestLinear:
    def test_weight_gradient(self):
        np.random.seed(2)
        layer = Linear(8, 4)
        x = np.random.randn(2, 3, 8)
        d_out = np.random.randn(2, 3, 4)

        layer.forward(x)
        layer.backward(d_out)

        def loss_fn():
            return np.sum(layer.forward(x) * d_out)

        num_dW = numerical_gradient(loss_fn, layer.W)
        np.testing.assert_allclose(layer.dW, num_dW, atol=1e-5)

    def test_bias_gradient(self):
        np.random.seed(3)
        layer = Linear(8, 4)
        x = np.random.randn(2, 3, 8)
        d_out = np.random.randn(2, 3, 4)

        layer.forward(x)
        layer.backward(d_out)

        def loss_fn():
            return np.sum(layer.forward(x) * d_out)

        num_db = numerical_gradient(loss_fn, layer.b)
        np.testing.assert_allclose(layer.db, num_db, atol=1e-5)

    def test_input_gradient(self):
        np.random.seed(4)
        layer = Linear(8, 4)
        x = np.random.randn(2, 3, 8)
        d_out = np.random.randn(2, 3, 4)

        layer.forward(x)
        dx = layer.backward(d_out)

        def loss_fn():
            return np.sum(layer.forward(x) * d_out)

        num_dx = numerical_gradient(loss_fn, x)
        np.testing.assert_allclose(dx, num_dx, atol=1e-5)


# ── LayerNorm gradient check ────────────────────────────────────────

class TestLayerNorm:
    def test_output_normalized(self):
        np.random.seed(5)
        ln = LayerNorm(16)
        x = np.random.randn(2, 4, 16) * 5 + 3
        out = ln.forward(x)
        # After layer norm, each vector should have near-zero mean and unit var
        means = out.mean(axis=-1)
        vrs = out.var(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(vrs, 1.0, atol=1e-2)

    def test_input_gradient(self):
        np.random.seed(6)
        ln = LayerNorm(8)
        x = np.random.randn(2, 3, 8)
        d_out = np.random.randn(2, 3, 8)

        ln.forward(x)
        dx = ln.backward(d_out)

        def loss_fn():
            return np.sum(ln.forward(x) * d_out)

        num_dx = numerical_gradient(loss_fn, x)
        np.testing.assert_allclose(dx, num_dx, atol=1e-5)


# ── Full FeedForwardBlock gradient check ─────────────────────────────

class TestFeedForwardBlock:
    def test_input_gradient(self):
        """Numerical gradient check on the full block (no dropout)."""
        np.random.seed(7)
        ffn = FeedForwardBlock(d_model=16, dropout_rate=0.0)
        x = np.random.randn(1, 2, 16)
        d_out = np.random.randn(1, 2, 16)

        ffn.forward(x, training=False)
        dx = ffn.backward(d_out)

        def loss_fn():
            return np.sum(ffn.forward(x, training=False) * d_out)

        num_dx = numerical_gradient(loss_fn, x)
        np.testing.assert_allclose(dx, num_dx, atol=1e-4)

    def test_residual_connection(self):
        """Output should differ from input only by the FFN contribution."""
        np.random.seed(8)
        ffn = FeedForwardBlock(d_model=16, dropout_rate=0.0)
        x = np.random.randn(1, 2, 16)
        out = ffn.forward(x, training=False)
        # With residual, output should not be zero even if FFN is small
        assert not np.allclose(out, 0.0)
        # But it also shouldn't be identical to x (FFN adds something)
        assert not np.allclose(out, x)

    def test_inference_deterministic(self):
        """With training=False, two forward passes should give identical output."""
        np.random.seed(9)
        ffn = FeedForwardBlock(d_model=16, dropout_rate=0.5)
        x = np.random.randn(1, 2, 16)
        out1 = ffn.forward(x, training=False)
        out2 = ffn.forward(x, training=False)
        np.testing.assert_array_equal(out1, out2)

    def test_parameter_count(self):
        ffn = FeedForwardBlock(d_model=32, expansion_factor=4)
        # ln: 32 (gamma) + 32 (beta) = 64
        # fc1: 32*128 + 128 = 4224
        # fc2: 128*32 + 32 = 4128
        # total = 64 + 4224 + 4128 = 8416
        assert ffn.num_parameters() == 8416


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
