"""
Feedforward (MLP) network as used in GPT-style transformer models.

Architecture (GPT-2 style):
    x -> LayerNorm -> Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model) -> Dropout -> + x
                                                                                                    ^
                                                                                          residual connection

This module implements the feedforward block with manual forward and backward
passes using NumPy, so the gradient flow is explicit and educational.
"""

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation (approximation used in GPT-2)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_backward(x: np.ndarray) -> np.ndarray:
    """Derivative of the GELU approximation with respect to its input."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner ** 2
    d_inner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner


class LayerNorm:
    """Layer normalisation (Ba et al., 2016)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)   # learnable scale
        self.beta = np.zeros(d_model)   # learnable shift

        # Gradient accumulators
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

        # Cache for backward pass
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : ndarray of shape (batch, seq_len, d_model)

        Returns
        -------
        out : same shape as x
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta

        self._cache = {"x_norm": x_norm, "std_inv": 1.0 / np.sqrt(var + self.eps),
                        "x_centered": x - mean}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        d_out : gradient flowing back, same shape as forward output

        Returns
        -------
        dx : gradient with respect to the input x
        """
        x_norm = self._cache["x_norm"]
        std_inv = self._cache["std_inv"]
        d_model = d_out.shape[-1]

        # Parameter gradients (summed over batch and sequence dimensions)
        self.d_gamma = (d_out * x_norm).sum(axis=tuple(range(d_out.ndim - 1)))
        self.d_beta = d_out.sum(axis=tuple(range(d_out.ndim - 1)))

        # Input gradient
        dx_norm = d_out * self.gamma
        dvar = (dx_norm * self._cache["x_centered"] * -0.5 * std_inv ** 3).sum(
            axis=-1, keepdims=True
        )
        dmean = (-dx_norm * std_inv).sum(axis=-1, keepdims=True) + dvar * (
            -2.0 * self._cache["x_centered"]
        ).mean(axis=-1, keepdims=True)

        dx = dx_norm * std_inv + dvar * 2.0 * self._cache["x_centered"] / d_model + dmean / d_model
        return dx


class Linear:
    """Fully-connected linear projection: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int):
        # Xavier / Glorot initialisation
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

        # Gradient accumulators
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : ndarray of shape (..., in_features)

        Returns
        -------
        out : ndarray of shape (..., out_features)
        """
        self._input = x
        return x @ self.W + self.b

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        d_out : gradient from upstream, shape (..., out_features)

        Returns
        -------
        dx : gradient with respect to x, shape (..., in_features)
        """
        x = self._input
        # Flatten leading dims for matmul, then reshape back
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        d_out_2d = d_out.reshape(-1, d_out.shape[-1])

        self.dW = x_2d.T @ d_out_2d
        self.db = d_out_2d.sum(axis=0)

        dx = d_out_2d @ self.W.T
        return dx.reshape(orig_shape)


class FeedForwardBlock:
    """
    GPT-style position-wise feedforward network.

    The standard architecture is:
        FFN(x) = Linear_2(GELU(Linear_1(x)))

    where Linear_1 projects from d_model to 4 * d_model (expansion) and
    Linear_2 projects back from 4 * d_model to d_model (contraction).

    A residual connection and layer normalisation wrap the block:
        output = x + Dropout(FFN(LayerNorm(x)))

    Parameters
    ----------
    d_model : int
        Dimensionality of the model (embedding size).
    expansion_factor : int
        Multiplicative factor for the hidden dimension (default 4 per GPT-2).
    dropout_rate : float
        Dropout probability applied after the second linear layer.
    """

    def __init__(self, d_model: int, expansion_factor: int = 4,
                 dropout_rate: float = 0.1):
        self.d_model = d_model
        d_hidden = d_model * expansion_factor

        self.ln = LayerNorm(d_model)
        self.fc1 = Linear(d_model, d_hidden)
        self.fc2 = Linear(d_hidden, d_model)

        self.dropout_rate = dropout_rate
        self._cache = {}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the feedforward block.

        Parameters
        ----------
        x : ndarray of shape (batch, seq_len, d_model)
            Input tensor (typically the output of the attention sub-layer).
        training : bool
            If True, dropout is applied; otherwise it is skipped.

        Returns
        -------
        out : ndarray, same shape as x
        """
        # --- pre-norm residual pathway (GPT-2 style) ---
        residual = x

        # 1. Layer normalisation
        h = self.ln.forward(x)

        # 2. First linear projection  (d_model -> 4*d_model)
        h = self.fc1.forward(h)

        # 3. GELU activation
        self._cache["pre_gelu"] = h          # needed for backward
        h = gelu(h)

        # 4. Second linear projection  (4*d_model -> d_model)
        h = self.fc2.forward(h)

        # 5. Dropout
        if training and self.dropout_rate > 0.0:
            mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(h.dtype)
            h = h * mask / (1.0 - self.dropout_rate)   # inverted dropout
            self._cache["dropout_mask"] = mask
        else:
            self._cache["dropout_mask"] = None

        # 6. Residual connection
        out = residual + h
        return out

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass — computes gradients for every learnable parameter
        and returns the gradient with respect to the block input.

        Parameters
        ----------
        d_out : ndarray of shape (batch, seq_len, d_model)
            Gradient flowing back from the downstream layer.

        Returns
        -------
        dx : ndarray, same shape as d_out
            Gradient with respect to the block input x.
        """
        # The residual connection splits the gradient into two paths
        d_residual = d_out          # path through the skip connection
        dh = d_out.copy()           # path through the FFN

        # 5. Dropout backward
        mask = self._cache["dropout_mask"]
        if mask is not None:
            dh = dh * mask / (1.0 - self.dropout_rate)

        # 4. Second linear backward
        dh = self.fc2.backward(dh)

        # 3. GELU backward
        dh = dh * gelu_backward(self._cache["pre_gelu"])

        # 2. First linear backward
        dh = self.fc1.backward(dh)

        # 1. LayerNorm backward
        dh = self.ln.backward(dh)

        # Combine residual path and FFN path
        dx = d_residual + dh
        return dx

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def parameters(self) -> list[tuple[str, np.ndarray]]:
        """Return a list of (name, parameter_array) pairs."""
        return [
            ("ln.gamma", self.ln.gamma),
            ("ln.beta", self.ln.beta),
            ("fc1.W", self.fc1.W),
            ("fc1.b", self.fc1.b),
            ("fc2.W", self.fc2.W),
            ("fc2.b", self.fc2.b),
        ]

    def gradients(self) -> list[tuple[str, np.ndarray]]:
        """Return a list of (name, gradient_array) pairs."""
        return [
            ("ln.gamma", self.ln.d_gamma),
            ("ln.beta", self.ln.d_beta),
            ("fc1.W", self.fc1.dW),
            ("fc1.b", self.fc1.db),
            ("fc2.W", self.fc2.dW),
            ("fc2.b", self.fc2.db),
        ]

    def num_parameters(self) -> int:
        """Total number of learnable scalar parameters."""
        return sum(p.size for _, p in self.parameters())


# ----------------------------------------------------------------------
# Quick demo / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    batch, seq_len, d_model = 2, 8, 64
    x = np.random.randn(batch, seq_len, d_model)

    ffn = FeedForwardBlock(d_model=d_model, dropout_rate=0.1)
    print(f"FeedForward block — {ffn.num_parameters():,} learnable parameters")
    print(f"  d_model      = {d_model}")
    print(f"  d_hidden     = {d_model * 4}")
    print(f"  input shape  = {x.shape}")

    # Forward
    out = ffn.forward(x, training=True)
    print(f"  output shape = {out.shape}")

    # Backward (use a dummy upstream gradient of ones)
    d_out = np.ones_like(out)
    dx = ffn.backward(d_out)
    print(f"  dx shape     = {dx.shape}")

    # Print gradient norms
    print("\n  Gradient norms:")
    for name, grad in ffn.gradients():
        print(f"    {name:12s}  {np.linalg.norm(grad):.6f}")
