---
name: llm-inference-energy
description: Reference data and methodology for LLM inference energy consumption (Wh, Joules, Wh per million tokens) across Claude, GPT, Gemini, Llama, etc., plus the relationship between sampling temperature and energy. Use when the user asks about LLM energy/power/carbon/water footprint per query or per token, the energy cost of high temperatures or reasoning modes, or wants comparative data for a dashboard, study, or article.
---

# LLM Inference Energy — Reference Notes

Snapshot of published estimates and methodology for LLM inference energy. Numbers vary by 1–2 orders of magnitude depending on hardware, batching, request size, and reasoning mode — always cite the source and assumptions.

## Headline Numbers (per-query, mixed input+output)

| Model | Energy / query | Source / assumption |
|---|---|---|
| GPT-4o | **~0.30 Wh** (typical short query, ~500 output tokens) | Epoch AI, Feb 2025. MoE w/ ~100B active params. |
| GPT-4o (range) | 0.42 ± 0.13 Wh | "How Hungry is AI?" arxiv 2505.09598 (2025) |
| ChatGPT (Altman) | **~0.34 Wh** "average prompt" | Sam Altman blog, 2025 — no methodology disclosed |
| Claude 3.7 Sonnet | **~0.81 Wh / query** | energycosts.co.uk synthesis, 2025 |
| Claude 3.5 Haiku | **~0.22 Wh / query** | same |
| BLOOM-176B (research) | ~4 Wh / query | Luccioni et al., 2022 — *no batching*, not representative of prod |
| Llama-65B | ~3–4 J / output token (longer generations) | TokenPowerBench, arxiv 2512.03024 |

## Per Million Tokens (Wh/M tok) — Derived

Most public estimates are per-query. To convert to Wh/M tok, divide by token count. Common derivations:

- **GPT-4o**: 0.30 Wh / 500 output tok ≈ **600 Wh / M output tok** (≈ 0.6 kWh/M)
- **Claude Sonnet (3.7) estimate**: ~390 Wh/M input, **~1,950 Wh/M output** (energycosts.co.uk, derived from pricing/hardware)
- **Claude Haiku**: roughly 4–5× lower than Sonnet on the same basis
- **GPT-3 era → H100 today**: ~120× improvement in J/token (two orders of magnitude). Implication: pre-2023 numbers are not useful for current dashboards.

> Rule of thumb for back-of-envelope: a modern flagship model on an H100 cluster lands in the **0.5–2 kWh per million output tokens** range; small/distilled models 0.05–0.5 kWh/M; "reasoning" / test-time-compute models can be 5–20× higher because they emit far more hidden tokens.

## Temperature ↔ Energy

The literature is thin here. Key findings:

1. **Per-token FLOPs are temperature-independent** at the sampler level. Temperature only rescales logits before `softmax`; it doesn't change the transformer forward pass cost.
2. **Indirect effect via output length**: higher temperature → more varied / sometimes longer outputs → more decoded tokens → more energy. Effect is small for `T ∈ [0, 1]` and grows non-monotonically toward `T ≥ 1.5`.
3. **Parallel sampling / self-consistency with temperature > 0**: ~5% extra power vs. greedy due to larger effective batch size (arxiv 2505.14733, "Energy Cost of Reasoning").
4. **Reasoning modes** (`reasoning_effort`, extended thinking, o1-style) dwarf any temperature effect — often **3–20×** baseline energy.
5. **Above T ≈ 2.0**, output token count can spike because the model rambles / loops before EOS — observed in informal benchmarks but not rigorously published.

**Modeling heuristic** (good enough for a dashboard):

```
E(T) ≈ E_base * (1 + alpha * T + beta * max(0, T - 1.5)^2)
```
with `alpha ≈ 0.05` (small linear bump from longer generations) and `beta ≈ 0.2` (super-linear blow-up past T=1.5 from incoherent / non-terminating output). Flag as "modeled, not measured" in UI.

> Note: most APIs cap temperature at 1.0 (Anthropic) or 2.0 (OpenAI). T=2.5 is extrapolation — useful for the "creative / hallucinating" end of a dashboard slider but rarely tested in production.

## Key Sources

Primary (peer-reviewed / institutional):
- Epoch AI — *How much energy does ChatGPT use?* https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use
- arxiv 2505.09598 — *How Hungry is AI? Benchmarking Energy, Water, Carbon Footprint of LLM Inference*
- arxiv 2505.14733 — *The Energy Cost of Reasoning: Test-time Compute*
- arxiv 2511.05597 — *From Prompts to Power: Measuring the Energy Footprint of LLM Inference*
- arxiv 2512.03024 — *TokenPowerBench*
- arxiv 2310.03003 — Luccioni et al., early per-query benchmark
- ScienceDirect S1364032125008329 — Systematic review of LLM electricity demand
- MIT Technology Review, 2025-05-20 — *We did the math on AI's energy footprint*

Secondary (synthesis / aggregators):
- llm-tracker.info — *Power Usage and Energy Efficiency*
- Simon Willison — tag *ai-energy-usage*
- energycosts.co.uk — Claude-specific estimates
- Hannah Ritchie substack — 2025 ChatGPT/Gemini carbon footprint
- Surfshark generative AI energy chart

## Caveats / Methodology Tips

- Always separate **input** vs **output** Wh/M tok — prefill and decode have different cost curves (prefill is FLOPs-bound, decode is memory-bandwidth-bound).
- "Per query" averages bake in assumed token counts (usually 500 out, 100 in). Always state the assumption.
- Vendor-published numbers (Altman's 0.34 Wh) are not auditable — flag as such.
- For carbon: multiply Wh by grid intensity (US avg ~400 gCO₂/kWh, Iceland ~30, India ~700).
- Water (cooling) is typically reported separately, ~5–50 mL per query for hyperscale DCs.

## Dashboard / Visualization Defaults

When building a temperature-vs-energy scatter:
- x-axis: temperature 0 → 2.5 (label that >1.0 or >2.0 is extrapolated for most vendors).
- y-axis: Wh per million **output** tokens, log scale if mixing flagship + nano models.
- Provide a "show modeled values" toggle — most non-T=default values are computed, not measured.
- Tooltip: `(T, Wh/Mtok, model, source-flag)`.
