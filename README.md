# gen-ai-sandbox

A NumPy-based implementation of the GPT-style feedforward (MLP) block with explicit forward and backward passes.

## Key Concepts

### LayerNorm (Layer Normalization)

LayerNorm solves a practical problem: as data flows through a deep network, the values at each layer can drift to wildly different scales (very large or very small), which makes training unstable. LayerNorm fixes this by **normalizing each individual vector to zero mean and unit variance**.

Given an input vector like `[2.0, 6.0, 4.0]`:

```
mean = (2 + 6 + 4) / 3 = 4.0
variance = average of squared deviations = 2.667
normalized = (x - mean) / sqrt(variance) = [-1.22, 1.22, 0.0]
```

Now the values are centered around 0 with a consistent spread. This happens independently for every token at every position in the sequence.

#### The problem with *only* normalizing

If we stopped here, we'd be forcing every layer's output into the same rigid distribution. That's too restrictive — sometimes the network *needs* values to be shifted or scaled differently to represent something useful. Pure normalization destroys information the network might want to keep.

#### Learnable scale (γ / gamma) and shift (β / beta)

These two small vectors give the network an escape hatch — a way to undo or adjust the normalization if that's what produces a lower loss:

```
output = γ * normalized + β
```

| Parameter | What it does | Initialized to |
|-----------|-------------|----------------|
| **γ (gamma)** — scale | Stretches or compresses each dimension | `[1, 1, 1, ...]` (no change initially) |
| **β (beta)** — shift | Moves each dimension up or down | `[0, 0, 0, ...]` (no change initially) |

They start as identity values (scale=1, shift=0) so initially LayerNorm just normalizes. During training, the optimizer adjusts γ and β via their gradients to find whatever scale and offset works best for each dimension.

#### Concrete example

```
input:        [2.0,  6.0,  4.0]
normalized:   [-1.22, 1.22, 0.0]

# After training, suppose the network learned:
γ = [0.5, 2.0, 1.0]
β = [0.1, 0.0, -0.3]

output = γ * normalized + β
       = [0.5*(-1.22)+0.1,  2.0*1.22+0.0,  1.0*0.0+(-0.3)]
       = [-0.51,            2.44,           -0.3]
```

Dimension 0 got compressed (γ=0.5), dimension 1 got amplified (γ=2.0), and dimension 2 got shifted down (β=-0.3). The network learned these values because they reduce the loss.

### Partial Derivatives: ∂L/∂x, ∂L/∂γ, ∂L/∂β

These are **partial derivatives** — the core building blocks of backpropagation. They describe how a small change in each variable affects the **loss** (L), which is the single number the model is trying to minimize during training.

`∂L/∂x` reads as "the partial derivative of L with respect to x" — or more intuitively, **"how much does the loss change if I nudge x a tiny bit?"**

#### In the context of LayerNorm

| Symbol | What it means | Why we need it |
|--------|--------------|----------------|
| **∂L/∂x** | How the loss changes w.r.t. the **input** to LayerNorm | Passed backward to the previous layer so it can update *its* parameters |
| **∂L/∂γ** | How the loss changes w.r.t. **gamma** (the learnable scale) | Used to update γ via gradient descent (e.g. `γ -= lr * ∂L/∂γ`) |
| **∂L/∂β** | How the loss changes w.r.t. **beta** (the learnable shift) | Used to update β via gradient descent (e.g. `β -= lr * ∂L/∂β`) |

#### Concrete example

Say during training the loss is `L = 4.2`. If `∂L/∂γ[0] = -0.3`, that means:

> "If I increase `γ[0]` by a tiny amount ε, the loss decreases by roughly `0.3 × ε`."

So the optimizer knows to **increase** `γ[0]` to reduce the loss.

#### The two roles in backprop

Every layer computes two kinds of gradients:

1. **Parameter gradients** (`∂L/∂γ`, `∂L/∂β`, `∂L/∂W`, `∂L/∂b`) — used to **update that layer's own weights**
2. **Input gradient** (`∂L/∂x`) — passed to the **previous layer** so the chain continues backward

The **chain rule** ties it all together — each layer only needs to know the gradient coming from above (`d_out`, which is `∂L/∂output`) and its own local computation to produce these three quantities.