# Mixture of Experts: How AI Models Scale Without Going Broke

*A visual deep-dive into sparse activation, gating networks, and the architecture powering modern large language models*

---

## The core idea

A standard dense neural network activates **all** of its parameters for every single input. That works fine at small scale, but once you're dealing with hundreds of billions of parameters, it becomes ruinously expensive — every forward pass touches everything.

Mixture of Experts (MoE) breaks this constraint. Instead of one monolithic network, you build **N specialist sub-networks** (the "experts"), plus a lightweight **gating network** that decides which 1 or 2 experts to actually use for each token. The rest stay dormant, saving compute.

> **The key insight:** You can have a model with 256 billion *total* parameters but only activate ~8 billion for any given token. You get the knowledge capacity of a huge model at the inference cost of a much smaller one.

---

## 🎮 Demo 1: The Gating Network

*See how a gating network routes different tokens to different specialists in real time.*

→ **[Open Demo: Gating Network Visualizer](./demo-1-gating.html)**

The demo lets you feed different token types (code, math, multilingual, logical, factual) into a simplified MoE layer and watch the routing weights update live. Notice how "def sort(arr):" reliably activates the code expert with ~72% weight, while "∫ f(x) dx" pivots strongly toward the math expert.

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

```
G(x) = Softmax( TopK(W_g · x, k) )

Output = Σᵢ G(x)ᵢ · Eᵢ(x)
```

Where `W_g` is the learned gate weight matrix, `TopK` selects only the k highest logits (setting the rest to −∞), and the output is a weighted sum of only the active experts.

Some implementations add Gaussian noise before TopK to encourage exploration during training:

```
H(x)ᵢ = (W_g · x)ᵢ + StandardNormal() · Softplus((W_noise · x)ᵢ)
```

This noise prevents the gating network from converging too quickly on a fixed set of favorites.

---

## 🎮 Demo 2: Sparse Activation Explorer

*Visualize how sparsity scales across a model's expert pool.*

→ **[Open Demo: Sparse Activation Explorer](./demo-2-sparsity.html)**

Drag the top-k slider from 1 to 16 across a 64-expert pool. Watch how the activation rate, compute savings, and effective capacity ratio change. At Top-2 you're activating just 3.1% of experts — using 96.9% less expert compute than a fully dense model.

---

## The load balancing problem

Here's the catch: if the gating network learns to always route to the same 2–3 "popular" experts, the others never get trained. You end up with a model that's effectively much smaller than its parameter count suggests. This is called **expert collapse**.

The fix is an **auxiliary load-balancing loss** added to the training objective. It penalizes routing distributions where some experts receive many more tokens than others, nudging the gating network toward even coverage.

Google's Switch Transformer defines the load balancing loss as:

```
L_aux = α · N · Σᵢ fᵢ · pᵢ
```

Where `fᵢ` is the fraction of tokens dispatched to expert *i*, `pᵢ` is the average routing probability for expert *i*, N is the number of experts, and α is a small hyperparameter (typically 0.01).

---

## 🎮 Demo 3: Load Balancing Simulator

*Train a tiny MoE and watch expert collapse happen — then fix it with the auxiliary loss.*

→ **[Open Demo: Load Balancing Simulator](./demo-3-load-balance.html)**

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
