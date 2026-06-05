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

<style>
.info-tip{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;border:1px solid #4a4860;color:#7a7890;font-size:9px;font-style:italic;font-family:Georgia,serif;cursor:default;position:relative;flex-shrink:0;user-select:none;vertical-align:middle;margin-left:5px;line-height:1;}
.info-tip:hover{border-color:#9b8dff;color:#9b8dff;}
.info-tip .tip-body{visibility:hidden;opacity:0;position:absolute;left:50%;bottom:calc(100% + 8px);transform:translateX(-50%);background:#1f1f28;border:1px solid #3a3a52;border-radius:8px;padding:10px 13px;font-size:12px;font-style:normal;font-family:-apple-system,'Segoe UI',sans-serif;color:#e8e6f0;white-space:normal;width:240px;line-height:1.6;z-index:200;box-shadow:0 6px 20px rgba(0,0,0,.5);pointer-events:none;transition:opacity .15s;}
.info-tip:hover .tip-body{visibility:visible;opacity:1;}
.info-tip .tip-body::after{content:'';position:absolute;top:100%;left:50%;transform:translateX(-50%);border:5px solid transparent;border-top-color:#3a3a52;}
.diagram-tips{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin-top:10px;font-size:12px;color:#6b6880;}
.diagram-tips span{display:flex;align-items:center;}
.info-tip .tip-body.tip-below{top:calc(100% + 8px) !important;bottom:auto !important;}
.info-tip .tip-body.tip-below::after{top:auto;bottom:100%;border-top-color:transparent;border-bottom-color:#3a3a52;}
.info-tip .tip-body.tip-left{left:0 !important;right:auto !important;transform:none !important;}
.info-tip .tip-body.tip-right{left:auto !important;right:0 !important;transform:none !important;}
</style>

<figure style="margin: 2em 0;">
<svg viewBox="0 0 920 390" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;font-family:-apple-system,'Segoe UI',sans-serif;background:#0f0f13;border-radius:12px;border:1px solid #2e2e3a;">
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

  <!-- Title -->
  <text x="460" y="26" text-anchor="middle" fill="#9090b0" font-size="12" letter-spacing="0.07em">MIXTURE OF EXPERTS — SINGLE LAYER FORWARD PASS</text>

  <!-- INPUT -->
  <rect x="16" y="158" width="124" height="62" rx="8" fill="#17171d" stroke="#9b8dff" stroke-width="1.5"/>
  <text x="78" y="182" text-anchor="middle" fill="#e8e6f0" font-size="14" font-weight="600">Token</text>
  <text x="78" y="205" text-anchor="middle" fill="#9b8dff" font-size="20" font-family="Georgia,serif" font-style="italic">x</text>

  <!-- Input → Gate -->
  <line x1="140" y1="189" x2="182" y2="189" stroke="#9b8dff" stroke-width="1.5" marker-end="url(#moe-arr-purple)"/>

  <!-- GATE — wider box, formula split across two short lines -->
  <rect x="184" y="130" width="192" height="120" rx="8" fill="#1a1628" stroke="#9b8dff" stroke-width="1.5"/>
  <text x="280" y="158" text-anchor="middle" fill="#e8e6f0" font-size="14" font-weight="600">Gating Network</text>
  <text x="280" y="179" text-anchor="middle" fill="#9b8dff" font-size="12" font-family="monospace">W_g · x  →  N logits</text>
  <text x="280" y="198" text-anchor="middle" fill="#9b8dff" font-size="12" font-family="monospace">TopK(k=2) + softmax</text>
  <text x="280" y="217" text-anchor="middle" fill="#8888a8" font-size="11">scores every expert</text>
  <text x="280" y="235" text-anchor="middle" fill="#8888a8" font-size="11">only top 2 activate</text>

  <!-- Gate → fan junction -->
  <line x1="376" y1="190" x2="408" y2="190" stroke="#6b6880" stroke-width="1" stroke-dasharray="4,3"/>
  <!-- Vertical fan bar -->
  <line x1="410" y1="42" x2="410" y2="350" stroke="#2e2e3a" stroke-width="1.5"/>

  <!-- EXPERTS: w=182 h=54 gap=10, start y=40 -->
  <!-- E1 center=67, E2 center=131, E3 center=195, E4 center=259, E5 center=323 -->

  <!-- EXPERT 1 (active, 72%) -->
  <rect x="448" y="40" width="182" height="54" rx="7" fill="#0b2218" stroke="#3ecfa4" stroke-width="2"/>
  <text x="539" y="63" text-anchor="middle" fill="#3ecfa4" font-size="13" font-weight="700">Expert 1 — Code</text>
  <text x="539" y="82" text-anchor="middle" fill="#3ecfa4" font-size="11">active · weight 0.72</text>
  <line x1="410" y1="67" x2="446" y2="67" stroke="#3ecfa4" stroke-width="2" marker-end="url(#moe-arr-teal)"/>
  <text x="428" y="60" text-anchor="middle" fill="#3ecfa4" font-size="12" font-weight="700">0.72</text>

  <!-- EXPERT 2 (inactive) -->
  <rect x="448" y="104" width="182" height="54" rx="7" fill="#141420" stroke="#3a3a52" stroke-width="1.5"/>
  <text x="539" y="127" text-anchor="middle" fill="#9090b0" font-size="13">Expert 2 — Language</text>
  <text x="539" y="146" text-anchor="middle" fill="#6b6880" font-size="11">dormant · weight ≈ 0</text>
  <line x1="410" y1="131" x2="446" y2="131" stroke="#3a3a52" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- EXPERT 3 (inactive) -->
  <rect x="448" y="168" width="182" height="54" rx="7" fill="#141420" stroke="#3a3a52" stroke-width="1.5"/>
  <text x="539" y="191" text-anchor="middle" fill="#9090b0" font-size="13">Expert 3 — Factual</text>
  <text x="539" y="210" text-anchor="middle" fill="#6b6880" font-size="11">dormant · weight ≈ 0</text>
  <line x1="410" y1="195" x2="446" y2="195" stroke="#3a3a52" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- EXPERT 4 (active, 28%) -->
  <rect x="448" y="232" width="182" height="54" rx="7" fill="#0b2218" stroke="#3ecfa4" stroke-width="2"/>
  <text x="539" y="255" text-anchor="middle" fill="#3ecfa4" font-size="13" font-weight="700">Expert 4 — Math</text>
  <text x="539" y="274" text-anchor="middle" fill="#3ecfa4" font-size="11">active · weight 0.28</text>
  <line x1="410" y1="259" x2="446" y2="259" stroke="#3ecfa4" stroke-width="2" marker-end="url(#moe-arr-teal)"/>
  <text x="428" y="252" text-anchor="middle" fill="#3ecfa4" font-size="12" font-weight="700">0.28</text>

  <!-- EXPERT 5 (inactive) -->
  <rect x="448" y="296" width="182" height="54" rx="7" fill="#141420" stroke="#3a3a52" stroke-width="1.5"/>
  <text x="539" y="319" text-anchor="middle" fill="#9090b0" font-size="13">Expert 5 — Logical</text>
  <text x="539" y="338" text-anchor="middle" fill="#6b6880" font-size="11">dormant · weight ≈ 0</text>
  <line x1="410" y1="323" x2="446" y2="323" stroke="#3a3a52" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- Active experts → Output (cubic bezier) -->
  <path d="M 630,67 C 668,67 668,172 694,172" stroke="#3ecfa4" stroke-width="1.5" fill="none" stroke-dasharray="5,3" marker-end="url(#moe-arr-teal)"/>
  <path d="M 630,259 C 668,259 668,204 694,204" stroke="#3ecfa4" stroke-width="1.5" fill="none" stroke-dasharray="5,3" marker-end="url(#moe-arr-teal)"/>

  <!-- OUTPUT -->
  <rect x="696" y="150" width="192" height="90" rx="8" fill="#1c1a10" stroke="#f5b942" stroke-width="1.5"/>
  <text x="792" y="176" text-anchor="middle" fill="#e8e6f0" font-size="14" font-weight="600">Output y</text>
  <text x="792" y="200" text-anchor="middle" fill="#f5b942" font-size="13" font-family="monospace">= 0.72 · E1(x)</text>
  <text x="792" y="222" text-anchor="middle" fill="#f5b942" font-size="13" font-family="monospace">+ 0.28 · E4(x)</text>

  <!-- Footer note -->
  <text x="460" y="372" text-anchor="middle" fill="#7a7a99" font-size="11">3 of 5 experts dormant — zero compute cost, full parameter capacity</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#6b6880;margin-top:10px;">A single MoE layer forward pass. The gating network scores all experts but only the top-k=2 fire. Dormant experts incur zero compute while still contributing to total model capacity.</figcaption>
<div class="diagram-tips">
  <span>Token x <span class="info-tip">i<span class="tip-body">The input token's embedding vector — a dense numerical representation of its meaning. This same vector is fed to both the gating network (to decide which experts activate) and the active experts themselves (to compute on).</span></span></span>
  <span>Gating Network <span class="info-tip">i<span class="tip-body">A single learned weight matrix W_g maps the token embedding to N scores (logits) — one per expert. TopK keeps only the k highest; softmax converts them to routing probabilities that sum to 1. The gate adds negligible compute overhead.</span></span></span>
  <span>Active experts <span class="info-tip">i<span class="tip-body">Only the top-k experts (teal border) receive the token and run their full feed-forward computation. The routing weights (0.72, 0.28) show each expert's contribution to the final output. Dormant experts contribute exactly zero for this token.</span></span></span>
  <span>Output y <span class="info-tip">i<span class="tip-body">A weighted sum of the active experts' output vectors: y = 0.72·E1(x) + 0.28·E4(x). Same shape as the input — the next transformer layer receives this as if it came from a single network, with no knowledge of the routing that produced it.</span></span></span>
</div>
</figure>

---

## Demo 1: The Gating Network

*See how a gating network routes different tokens to different specialists in real time.*

<iframe src="/files-4/demo-1-gating.html" style="width:100%; height:600px; border:none; border-radius:8px; margin: 1.5em 0; display:block;" loading="lazy" title="MoE Gating Network Visualizer"></iframe>

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

<iframe src="/files-4/demo-2-sparsity.html" style="width:100%; height:600px; border:none; border-radius:8px; margin: 1.5em 0; display:block;" loading="lazy" title="MoE Sparse Activation Explorer"></iframe>

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

<iframe src="/files-4/demo-3-load-balance.html" style="width:100%; height:600px; border:none; border-radius:8px; margin: 1.5em 0; display:block;" loading="lazy" title="MoE Load Balancing Simulator"></iframe>

Toggle the auxiliary loss on and off during a simulated training run. Without it, two or three experts quickly absorb nearly all the routing traffic. With it, the distribution stays healthy. You can also adjust the loss coefficient α to see how aggressively it corrects imbalance.

---

## Sparse vs. dense: the tradeoffs

<div style="margin: 2em 0; border-radius: 12px; overflow: hidden; border: 1px solid #2e2e3a; font-size: 13px; line-height: 1.5;">
<table style="width:100%; border-collapse: collapse; background: #0f0f13;">
  <colgroup>
    <col style="width:24%">
    <col style="width:38%">
    <col style="width:38%">
  </colgroup>
  <thead>
    <tr style="background:#17171d; border-bottom: 2px solid #2e2e3a;">
      <th style="padding:13px 18px; text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#6b6880; font-weight:600;">Dimension</th>
      <th style="padding:13px 18px; text-align:left; color:#e8e6f0; font-weight:600;">Dense model</th>
      <th style="padding:13px 18px; text-align:left; color:#9b8dff; font-weight:600;">MoE model</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#13131a;">
      <td colspan="3" style="padding:7px 18px; font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#4a4860; font-weight:600;">Compute</td>
    </tr>
    <tr style="border-bottom:1px solid #1e1e28;">
      <td style="padding:13px 18px; color:#e8e6f0; font-weight:600;">Training cost</td>
      <td style="padding:13px 18px; color:#3ecfa4;">✓&nbsp; Lower for same parameter count</td>
      <td style="padding:13px 18px; color:#f07a54;">↑&nbsp; Higher — all experts need gradient signal</td>
    </tr>
    <tr style="border-bottom:1px solid #1e1e28;">
      <td style="padding:13px 18px; color:#e8e6f0; font-weight:600;">Inference compute</td>
      <td style="padding:13px 18px; color:#6b6880;">Full params touched every token</td>
      <td style="padding:13px 18px; color:#3ecfa4;">✓&nbsp; Only active experts — 2–8× cheaper</td>
    </tr>
    <tr style="background:#13131a;">
      <td colspan="3" style="padding:7px 18px; font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#4a4860; font-weight:600;">Memory</td>
    </tr>
    <tr style="border-bottom:1px solid #1e1e28;">
      <td style="padding:13px 18px; color:#e8e6f0; font-weight:600;">VRAM footprint</td>
      <td style="padding:13px 18px; color:#3ecfa4;">✓&nbsp; Matches active parameter count</td>
      <td style="padding:13px 18px; color:#f07a54;">↑&nbsp; Must load all N experts into VRAM</td>
    </tr>
    <tr style="background:#13131a;">
      <td colspan="3" style="padding:7px 18px; font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#4a4860; font-weight:600;">Routing</td>
    </tr>
    <tr style="border-bottom:1px solid #1e1e28;">
      <td style="padding:13px 18px; color:#e8e6f0; font-weight:600;">Routing overhead</td>
      <td style="padding:13px 18px; color:#6b6880;">—&nbsp; None</td>
      <td style="padding:13px 18px; color:#6b6880;">≈&nbsp; Tiny — gating matrix only</td>
    </tr>
    <tr style="border-bottom:1px solid #1e1e28;">
      <td style="padding:13px 18px; color:#e8e6f0; font-weight:600;">Expert collapse</td>
      <td style="padding:13px 18px; color:#6b6880;">—&nbsp; Not applicable</td>
      <td style="padding:13px 18px; color:#f5b942;">⚠&nbsp; Real risk without load-balancing loss</td>
    </tr>
    <tr style="background:#13131a;">
      <td colspan="3" style="padding:7px 18px; font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:#4a4860; font-weight:600;">Hardware</td>
    </tr>
    <tr>
      <td style="padding:13px 18px; color:#e8e6f0; font-weight:600;">Distributed complexity</td>
      <td style="padding:13px 18px; color:#3ecfa4;">✓&nbsp; Simple all-reduce</td>
      <td style="padding:13px 18px; color:#f07a54;">↑&nbsp; Expert parallelism + all-to-all comms</td>
    </tr>
  </tbody>
</table>
</div>

The practical upshot: MoE models are memory-hungry but compute-efficient. They're best suited for **high-throughput inference** environments where you have lots of VRAM but want fast per-token latency.

### Expert parallelism: the hardware reality

The "expert parallelism + all-to-all comms" row in the table above is where MoE earns its reputation for being hard to serve. In practice, a model with 64 experts can't fit all of them on a single GPU, so experts are sharded across a group of GPUs — each device owns a subset. When a batch of tokens arrives, the gating network decides which expert each token goes to, then the system must physically move each token to the GPU that owns its assigned expert. This is the **all-to-all** step: every GPU sends some tokens to every other GPU before computation can begin, and then the results come back the same way.

That communication cost is the main reason you don't just swap a dense model for a MoE of the same active-parameter count and call it free. At scale, all-to-all across hundreds of GPUs becomes a meaningful fraction of step time — enough that DeepSeek-V3 describes minimizing it as a core design constraint, and why most MoE deployments require specialized parallelism libraries (Megablocks, DeepSpeed-MoE, or custom kernels) rather than off-the-shelf training loops. The routing problem and the communication problem are inseparable once you leave a single device.

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

- Fedus, W., Zoph, B., Shazeer, N. (2021). [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.*](https://arxiv.org/abs/2101.03961) — The foundational modern MoE paper.
- Jiang, A. Q. et al. (2024). [*Mixtral of Experts.*](https://arxiv.org/abs/2401.04088) — Practical open-weight MoE at 8×7B scale.
- Shazeer, N. et al. (2017). [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.*](https://arxiv.org/abs/1701.06538) — The original modern MoE proposal.
- DeepSeek-AI (2024). [*DeepSeek-V3 Technical Report.*](https://arxiv.org/abs/2412.19437) — State-of-the-art MoE with novel load-balancing approach.

---

*The interactive demos in this post are built with vanilla JavaScript and run entirely in the browser with no dependencies.*

<script>
window.addEventListener('message', function(e) {
  if (!e.data || typeof e.data.moeH !== 'number') return;
  var iframes = document.querySelectorAll('iframe');
  for (var i = 0; i < iframes.length; i++) {
    try {
      if (iframes[i].contentWindow === e.source) {
        iframes[i].style.height = e.data.moeH + 'px';
        break;
      }
    } catch (x) {}
  }
});
(function(){
  document.querySelectorAll('.info-tip').forEach(function(el){
    el.addEventListener('mouseenter', function(){
      var b = el.querySelector('.tip-body');
      if(!b) return;
      b.classList.remove('tip-below','tip-left','tip-right');
      var r = el.getBoundingClientRect();
      if(r.top < 160) b.classList.add('tip-below');
      var cx = r.left + r.width / 2;
      if(cx - 120 < 8) b.classList.add('tip-left');
      else if(cx + 120 > window.innerWidth - 8) b.classList.add('tip-right');
    });
  });
})();
</script>
