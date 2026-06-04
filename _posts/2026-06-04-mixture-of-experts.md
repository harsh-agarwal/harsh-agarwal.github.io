---
layout: playground-post
title: "Mixture of Experts: How AI Models Scale Without Going Broke"
date: 2026-06-04
---

*A visual deep-dive into sparse activation, gating networks, and the architecture powering modern large language models*

---

## The core idea

A standard dense neural network activates **all** of its parameters for every single input. That works fine at small scale, but once you're dealing with hundreds of billions of parameters, it becomes ruinously expensive — every forward pass touches everything.

Mixture of Experts (MoE) breaks this constraint. Instead of one monolithic network, you build **N specialist sub-networks** (the "experts"), plus a lightweight **gating network** that decides which 1 or 2 experts to actually use for each token. The rest stay dormant, saving compute.

> **The key insight:** You can have a model with 256 billion *total* parameters but only activate ~8 billion for any given token. You get the knowledge capacity of a huge model at the inference cost of a much smaller one.

<figure style="margin: 2em 0;">
<svg viewBox="0 0 800 350" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;font-family:-apple-system,'Segoe UI',sans-serif;background:#0f0f13;border-radius:12px;border:1px solid #2e2e3a;">
  <defs>
    <marker id="moe-arr-purple" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0.5 L0,5.5 L7,3 z" fill="#9b8dff"/>
    </marker>
    <marker id="moe-arr-teal" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0.5 L0,5.5 L7,3 z" fill="#3ecfa4"/>
    </marker>
    <marker id="moe-arr-amber" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0.5 L0,5.5 L7,3 z" fill="#f5b942"/>
    </marker>
  </defs>

  <text x="400" y="22" text-anchor="middle" fill="#6b6880" font-size="11" letter-spacing="0.08em">MIXTURE OF EXPERTS — SINGLE LAYER FORWARD PASS</text>

  <!-- INPUT -->
  <rect x="18" y="148" width="108" height="54" rx="8" fill="#17171d" stroke="#9b8dff" stroke-width="1.5"/>
  <text x="72" y="170" text-anchor="middle" fill="#e8e6f0" font-size="13" font-weight="600">Token</text>
  <text x="72" y="190" text-anchor="middle" fill="#9b8dff" font-size="19" font-family="Georgia,serif" font-style="italic">x</text>

  <!-- Input → Gate -->
  <line x1="126" y1="175" x2="172" y2="175" stroke="#9b8dff" stroke-width="1.5" marker-end="url(#moe-arr-purple)"/>

  <!-- GATE -->
  <rect x="174" y="128" width="148" height="94" rx="8" fill="#1a1628" stroke="#9b8dff" stroke-width="1.5"/>
  <text x="248" y="153" text-anchor="middle" fill="#e8e6f0" font-size="12" font-weight="600">Gating Network</text>
  <text x="248" y="172" text-anchor="middle" fill="#9b8dff" font-size="11" font-family="monospace">softmax( TopK(W_g · x) )</text>
  <text x="248" y="191" text-anchor="middle" fill="#6b6880" font-size="10">scores all N experts</text>
  <text x="248" y="208" text-anchor="middle" fill="#6b6880" font-size="10">activates only top k = 2</text>

  <!-- Gate → fan junction -->
  <line x1="322" y1="175" x2="362" y2="175" stroke="#6b6880" stroke-width="1" stroke-dasharray="4,3"/>
  <!-- Vertical fan bar -->
  <line x1="364" y1="45" x2="364" y2="305" stroke="#2e2e3a" stroke-width="1.5"/>

  <!-- EXPERT 1 (active, 72%) -->
  <rect x="404" y="43" width="148" height="44" rx="6" fill="#0b2218" stroke="#3ecfa4" stroke-width="2"/>
  <text x="478" y="62" text-anchor="middle" fill="#3ecfa4" font-size="12" font-weight="700">Expert 1 — Code</text>
  <text x="478" y="78" text-anchor="middle" fill="#3ecfa4" font-size="10" opacity="0.8">active · weight 0.72</text>
  <line x1="364" y1="65" x2="402" y2="65" stroke="#3ecfa4" stroke-width="2" marker-end="url(#moe-arr-teal)"/>
  <text x="383" y="59" text-anchor="middle" fill="#3ecfa4" font-size="11" font-weight="700">0.72</text>

  <!-- EXPERT 2 (inactive) -->
  <rect x="404" y="98" width="148" height="44" rx="6" fill="#17171d" stroke="#2e2e3a" stroke-width="1.2" opacity="0.45"/>
  <text x="478" y="117" text-anchor="middle" fill="#6b6880" font-size="12">Expert 2 — Language</text>
  <text x="478" y="133" text-anchor="middle" fill="#3e3e4e" font-size="10">dormant · weight ≈ 0</text>
  <line x1="364" y1="120" x2="402" y2="120" stroke="#2e2e3a" stroke-width="1" stroke-dasharray="3,3"/>

  <!-- EXPERT 3 (inactive) -->
  <rect x="404" y="153" width="148" height="44" rx="6" fill="#17171d" stroke="#2e2e3a" stroke-width="1.2" opacity="0.45"/>
  <text x="478" y="172" text-anchor="middle" fill="#6b6880" font-size="12">Expert 3 — Factual</text>
  <text x="478" y="188" text-anchor="middle" fill="#3e3e4e" font-size="10">dormant · weight ≈ 0</text>
  <line x1="364" y1="175" x2="402" y2="175" stroke="#2e2e3a" stroke-width="1" stroke-dasharray="3,3"/>

  <!-- EXPERT 4 (active, 28%) -->
  <rect x="404" y="208" width="148" height="44" rx="6" fill="#0b2218" stroke="#3ecfa4" stroke-width="2"/>
  <text x="478" y="227" text-anchor="middle" fill="#3ecfa4" font-size="12" font-weight="700">Expert 4 — Math</text>
  <text x="478" y="243" text-anchor="middle" fill="#3ecfa4" font-size="10" opacity="0.8">active · weight 0.28</text>
  <line x1="364" y1="230" x2="402" y2="230" stroke="#3ecfa4" stroke-width="2" marker-end="url(#moe-arr-teal)"/>
  <text x="383" y="224" text-anchor="middle" fill="#3ecfa4" font-size="11" font-weight="700">0.28</text>

  <!-- EXPERT 5 (inactive) -->
  <rect x="404" y="263" width="148" height="44" rx="6" fill="#17171d" stroke="#2e2e3a" stroke-width="1.2" opacity="0.45"/>
  <text x="478" y="282" text-anchor="middle" fill="#6b6880" font-size="12">Expert 5 — Logical</text>
  <text x="478" y="298" text-anchor="middle" fill="#3e3e4e" font-size="10">dormant · weight ≈ 0</text>
  <line x1="364" y1="285" x2="402" y2="285" stroke="#2e2e3a" stroke-width="1" stroke-dasharray="3,3"/>

  <!-- Active experts → Output (cubic bezier curves) -->
  <path d="M 552,65 C 592,65 592,162 628,162" stroke="#3ecfa4" stroke-width="1.5" fill="none" stroke-dasharray="5,3" marker-end="url(#moe-arr-teal)"/>
  <path d="M 552,230 C 592,230 592,192 628,192" stroke="#3ecfa4" stroke-width="1.5" fill="none" stroke-dasharray="5,3" marker-end="url(#moe-arr-teal)"/>

  <!-- OUTPUT -->
  <rect x="630" y="142" width="152" height="70" rx="8" fill="#1c1a10" stroke="#f5b942" stroke-width="1.5"/>
  <text x="706" y="164" text-anchor="middle" fill="#e8e6f0" font-size="12" font-weight="600">Output y</text>
  <text x="706" y="182" text-anchor="middle" fill="#f5b942" font-size="11" font-family="monospace">0.72 · E₁(x)</text>
  <text x="706" y="198" text-anchor="middle" fill="#f5b942" font-size="11" font-family="monospace">+ 0.28 · E₄(x)</text>

  <!-- Footer note -->
  <text x="478" y="326" text-anchor="middle" fill="#3e3e4e" font-size="10">3 of 5 experts dormant — zero compute cost, full parameter capacity</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#6b6880;margin-top:10px;">A single MoE layer forward pass. The gating network scores all experts but only the top-k=2 fire. Dormant experts incur zero compute while still contributing to total model capacity.</figcaption>
</figure>

---

## Demo 1: The Gating Network

*See how a gating network routes different tokens to different specialists in real time.*

<iframe src="/files-4/demo-1-gating.html" style="width:100%; height:700px; border:none; border-radius:8px; margin: 1.5em 0; display:block;" loading="lazy" title="MoE Gating Network Visualizer"></iframe>

Feed different token types (code, math, multilingual, logical, factual) into a simplified MoE layer and watch the routing weights update live. Notice how `def sort(arr):` reliably activates the code expert with ~72% weight, while `∫ f(x) dx` pivots strongly toward the math expert.

---

## How it works, step by step

### 1. Token arrives as an embedding

Each input token is converted to a high-dimensional vector. This vector carries the token's meaning and context from previous layers — it's the same embedding used in any transformer. What comes next is what makes MoE different.

### 2. The gating network scores each expert

A small linear layer `W_g` is multiplied by the token embedding to produce one logit per expert. If you have 64 experts, you get a vector of 64 scores. The gating network is tiny — just one weight matrix — so it adds negligible overhead.

### 3. TopK selects the winners

Only the top-k logit positions are kept (typically k=1 or k=2). All other logits are set to −∞ before the softmax step. This is the sparsity mechanism: a hard selection that routes the token to just a few experts and skips the rest entirely.

### 4. Softmax converts logits to weights

The remaining k logits pass through softmax to produce routing probabilities that sum to 1.0. If k=2 and the scores are 0.78 and 0.22, the token will be processed by both experts with those proportional contributions.

### 5. Experts compute in parallel

The selected experts — each a full feed-forward network, typically a large MLP — process the token representation independently and in parallel. On real hardware this dispatch is done via all-to-all communication across GPU nodes.

### 6. Weighted sum produces the output

Expert outputs are multiplied by their routing weights and summed:

```
Output = 0.78 · E₁(x) + 0.22 · E₂(x)
```

The result is a single vector with the same shape as the input, ready for the next transformer layer. The whole pipeline is fully differentiable; gradients flow through the gating weights and both active experts.

---

## The math

The gating function is elegantly simple. For an input *x* and expert networks *E₁…Eₙ*:

$$G(x) = \text{Softmax}\!\left(\text{TopK}(W_g \cdot x,\ k)\right)$$

$$\text{Output} = \sum_i G(x)_i \cdot E_i(x)$$

Where `W_g` is the learned gate weight matrix, `TopK` selects only the k highest logits (setting the rest to −∞), and the output is a weighted sum of only the active experts.

Some implementations add Gaussian noise before TopK to encourage exploration during training:

$$H(x)_i = (W_g \cdot x)_i + \mathcal{N}(0,1) \cdot \text{Softplus}\!\left((W_\text{noise} \cdot x)_i\right)$$

This noise prevents the gating network from converging too quickly on a fixed set of favorites.

---

## Demo 2: Sparse Activation Explorer

*Visualize how sparsity scales across a model's expert pool.*

<iframe src="/files-4/demo-2-sparsity.html" style="width:100%; height:680px; border:none; border-radius:8px; margin: 1.5em 0; display:block;" loading="lazy" title="MoE Sparse Activation Explorer"></iframe>

Drag the top-k slider from 1 to 16 across a 64-expert pool. Watch how the activation rate, compute savings, and effective capacity ratio change. At Top-2 you're activating just 3.1% of experts — using 96.9% less expert compute than a fully dense model.

---

## The load balancing problem

Here's the catch: if the gating network learns to always route to the same 2–3 "popular" experts, the others never get trained. You end up with a model that's effectively much smaller than its parameter count suggests. This is called **expert collapse**.

The fix is an **auxiliary load-balancing loss** added to the training objective. It penalizes routing distributions where some experts receive many more tokens than others, nudging the gating network toward even coverage.

Google's Switch Transformer defines the load balancing loss as:

$$\mathcal{L}_\text{aux} = \alpha \cdot N \cdot \sum_i f_i \cdot p_i$$

Where $f_i$ is the fraction of tokens dispatched to expert $i$, $p_i$ is the average routing probability for expert $i$, $N$ is the number of experts, and $\alpha$ is a small hyperparameter (typically 0.01).

---

## Demo 3: Load Balancing Simulator

*Train a tiny MoE and watch expert collapse happen — then fix it with the auxiliary loss.*

<iframe src="/files-4/demo-3-load-balance.html" style="width:100%; height:760px; border:none; border-radius:8px; margin: 1.5em 0; display:block;" loading="lazy" title="MoE Load Balancing Simulator"></iframe>

Toggle the auxiliary loss on and off during a simulated training run. Without it, two or three experts quickly absorb nearly all the routing traffic. With it, the distribution stays healthy. You can also adjust the loss coefficient α to see how aggressively it corrects imbalance.

---

## Sparse vs. dense: the tradeoffs

| Dimension | Dense model | MoE model |
|---|---|---|
| **Training cost** | Lower for same param count | Higher — all experts need gradient signal |
| **Inference compute** | Proportional to all params | Only active experts (2–8× cheaper) |
| **Memory footprint** | Matches active params | Must load all experts into VRAM |
| **Routing overhead** | None | Small — gating network is tiny |
| **Expert collapse** | Not applicable | Real risk without load-balancing loss |
| **Hardware complexity** | Simple all-reduce | Expert parallelism + all-to-all comms |

The practical upshot: MoE models are memory-hungry but compute-efficient. They're best suited for **high-throughput inference** environments where you have lots of VRAM but want fast per-token latency.

---

## Where you'll find MoE in the wild

**Switch Transformer (Google, 2021)** — The paper that proved MoE could scale to trillions of parameters. Used Top-1 routing for simplicity and showed that even with the instability it introduces, the efficiency gains were worth it.

**Mixtral 8×7B (Mistral AI, 2023)** — The most prominent open-weight MoE model. 8 experts per layer, Top-2 routing, ~13B active parameters out of 46B total. Performance competitive with dense models at 2–3× the size.

**GPT-4 (OpenAI, 2023)** — Widely believed to use MoE architecture based on leaks and inference from its efficiency characteristics. OpenAI has never confirmed the architecture details.

**DeepSeek-V3 (DeepSeek, 2024)** — Uses fine-grained expert segmentation and an auxiliary-loss-free load balancing strategy. 671B total parameters, 37B active per token, trained at remarkably low cost.

---

## Key takeaways

1. **MoE decouples capacity from compute.** Total parameters determine what the model *knows*; active parameters determine how much it *thinks* per token.

2. **The gating network is the heart of the system.** A small linear layer + TopK + softmax is all it takes to route tokens to specialists. The elegance is in the simplicity.

3. **Load balancing is the central training challenge.** Expert collapse is real and subtle — the auxiliary loss is not optional in practice.

4. **Memory is the bottleneck, not compute.** All experts must be loaded into VRAM even though most are idle at any moment. This shapes what hardware MoE models run well on.

5. **MoE is increasingly the default at scale.** The efficiency gains are too significant to ignore once model size crosses ~10B parameters. Expect it everywhere.

---

## Further reading

- Fedus, W., Zoph, B., Shazeer, N. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* — The foundational modern MoE paper.
- Jiang, A. Q. et al. (2024). *Mixtral of Experts.* — Practical open-weight MoE at 8×7B scale.
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* — The original modern MoE proposal.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* — State-of-the-art MoE with novel load-balancing approach.

---

*The interactive demos in this post are built with vanilla JavaScript and run entirely in the browser with no dependencies.*
