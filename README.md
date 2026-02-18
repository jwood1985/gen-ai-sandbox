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

### Model Dimensions: Parameter Count, Layer Count, and d_model

These three numbers are the primary knobs that determine a model's size and capability. They are related but measure fundamentally different things.

#### d_model (model dimension / embedding size)

This is the **width** of the model — the size of the vector used to represent each token at every point in the network. Every token in the input sequence gets converted into a vector of exactly `d_model` numbers, and that vector flows through every layer.

Think of it as **how much information the model can carry about a single token at once**. A larger d_model means each token's representation has more room to encode nuances of meaning, context, and relationships.

```
"The cat sat" with d_model=4:

"The" → [0.21, -0.87, 0.45, 0.12]
"cat" → [0.93,  0.11, -0.34, 0.67]
"sat" → [-0.15, 0.52, 0.88, -0.41]
```

With d_model=768 (GPT-2 Small), each token carries 768 numbers — far more expressive, but also far more parameters in every weight matrix that touches those vectors.

#### Layer count (depth)

This is the **depth** of the model — how many transformer blocks are stacked on top of each other. Each layer is one complete transformer block containing both a multi-head attention sub-layer and a feedforward block (like the `FeedForwardBlock` in `feedforward.py`).

Think of it as **how many sequential processing steps the model gets**. Each layer can refine, combine, and transform the token representations produced by the previous layer:

```
Input embeddings
  → Layer 1:  basic pattern recognition (syntax, common phrases)
  → Layer 2:  combining nearby tokens
  → ...
  → Layer 12: higher-level understanding (semantics, reasoning)
  → Output
```

More layers means the model can build up more abstract, complex representations — but each layer adds more parameters and compute.

#### Parameter count (total learnable weights)

This is the **total number of individual numbers the model must learn during training**. It is a consequence of d_model, layer count, and other architecture choices — not an independent knob.

For a single feedforward block, the parameter count comes from:

```
LayerNorm:  d_model (γ) + d_model (β)                          = 2 * d_model
Linear 1:   d_model * 4*d_model (W) + 4*d_model (b)            = 4*d_model² + 4*d_model
Linear 2:   4*d_model * d_model (W) + d_model (b)              = 4*d_model² + d_model
                                                        Total  ≈ 8 * d_model²
```

So for d_model=64 (as in `feedforward.py`), one block has ~33K parameters. For d_model=768, one block has ~4.7M parameters. Multiply by the number of layers (and add attention parameters) to get the full model count.

#### How they relate

| | d_model | Layers | Parameters |
|--|---------|--------|------------|
| **What it controls** | Width of token representations | Depth of processing | Total learnable weights |
| **Analogy** | How detailed each note on a page is | How many pages of revision you get | Total ink used across all pages |
| **Increasing it** | Richer per-token information | More processing steps | More capacity, more compute |

#### Reference models

| Model | d_model | Layers | Parameters |
|-------|---------|--------|------------|
| GPT-2 Small | 768 | 12 | 117M |
| GPT-2 Medium | 1024 | 24 | 345M |
| GPT-2 Large | 1280 | 36 | 774M |
| GPT-2 XL | 1600 | 48 | 1.5B |
| GPT-3 | 12288 | 96 | 175B |
| LLaMA 7B | 4096 | 32 | 7B |
| LLaMA 70B | 8192 | 80 | 70B |

Notice the pattern: as models get bigger, **both** d_model and layer count increase. The parameter count grows roughly as `layers * d_model²`, so doubling d_model quadruples the parameters per layer.