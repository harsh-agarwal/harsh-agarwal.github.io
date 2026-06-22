---
layout: playground-post
title: "From Next-Token Prediction to Iterative Unmasking: A Visual Guide to Text Diffusion"
date: 2026-06-22
description: "Why text diffusion is a genuinely different way to think about language generation, and how 200 lines of PyTorch can show you the entire idea."
tags: [diffusion, nlp, pytorch, language-models]
---

GPT writes left to right, one token at a time, no going back. Gemma Diffusion starts from a fully masked canvas and thinks its way to fluency all at once. Same transformer backbone, completely different bet about what generation should look like. This post visualizes every step of the idea, grounded in a 200-line PyTorch implementation you can run yourself.

<style>
.fig-note {
  font-size: 13px;
  color: #777;
  font-style: italic;
  margin: -4px 0 20px;
  line-height: 1.6;
}
.fig-iframe {
  border: 1px solid #e8e8e8;
  border-radius: 10px;
  width: 100%;
  display: block;
  margin: 1.5rem 0 0.5rem;
  overflow: hidden;
}
.comparison-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin: 16px 0;
}
.comparison-table th {
  background: #F4F6FB;
  color: #3B4562;
  font-weight: 600;
  text-align: left;
  padding: 9px 12px;
  border: 1px solid #DDE3ED;
}
.comparison-table td {
  padding: 8px 12px;
  border: 1px solid #DDE3ED;
  vertical-align: top;
}
.comparison-table tr:nth-child(even) { background: #FAFBFD; }
blockquote {
  border-left: 3px solid #c47a2a;
  margin: 24px 0;
  padding: 12px 20px;
  background: #fffbf4;
  border-radius: 0 6px 6px 0;
  color: #555;
  font-style: italic;
}
</style>

---

## 1. How language models generate text today

Modern LLMs are **autoregressive** models. They generate text one token at a time, left to right, each new token conditioned on everything before it. The process is conceptually simple: predict the next token, append it, repeat.

This works extraordinarily well. But the sequential commitment bakes in structural constraints worth understanding.

<iframe src="/playgrounds/text-diffusion/fig4-llm-vs-diffusion-demo.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="Interactive demo: autoregressive vs diffusion generation"></iframe>
<p class="fig-note">Try the <strong>Autoregressive</strong> tab. Each row locks in a token permanently; the model commits to every choice without knowing what comes next. Then switch to <strong>Text diffusion</strong> to see all positions fill in parallel.</p>

The sequential nature creates three concrete limits:

- **Early commitments are irreversible.** The model writes "The president of France is..." and must pick the next word before seeing how the sentence ends. There is no mechanism to revise that choice.
- **Generation scales linearly with length.** Generating 200 tokens requires 200 sequential forward passes. There is no way to parallelize across tokens.
- **No natural refinement loop.** LLMs have no built-in way to "think again" about an early choice. What's generated is what's kept.

---

## 2. What does "noise" mean for discrete tokens?

Image diffusion defines noise as Gaussian perturbation: add random continuous values to pixel intensities until the image is pure static, then train a network to reverse the process. Text can't work this way. Token IDs are discrete integers; there is no sensible notion of adding 0.3 to the token "cat."

The natural analog is **masking**: replace tokens with a special `[MASK]` symbol. At full noise, every token is masked. At zero noise, the text is clean. The model's job is to fill in the blanks, using context from every other position simultaneously.

<iframe src="/playgrounds/text-diffusion/fig1-forward-process.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="Figure 1: forward process noise slider"></iframe>
<p class="fig-note">Drag the slider. Each position masks independently with probability t/T, so you can jump to any noise level in a single step without simulating all steps in between. This independence is what makes training efficient.</p>

The forward process is:

```python
T = 20  # total diffusion timesteps

def mask_rate(t: int) -> float:
    return t / T

def q_forward(x0: torch.Tensor, t: int) -> torch.Tensor:
    corrupted = x0.clone()
    should_mask = torch.rand(x0.shape) < mask_rate(t)
    corrupted[should_mask] = MASK_ID  # index 27
    return corrupted
```

A linear schedule: at `t=0`, no masking; at `t=T`, every token is `[MASK]`. Each position is corrupted independently, with no memory between positions or across timesteps.

---

## 3. The model: three signals, one objective

The denoising model takes a partially-masked sequence and predicts the original tokens at every position simultaneously. To do this well it needs to know three things: what token is currently at each position, where in the sequence that position is, and how much corruption was applied.

<iframe src="/playgrounds/text-diffusion/fig2-architecture.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="Figure 2: model architecture diagram"></iframe>
<p class="fig-note">Hover over each component for details. The key structural choice is the bidirectional transformer: no causal mask means every position can attend to every other position, in both directions.</p>

```python
class DenoisingTransformer(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.tok_emb = nn.Embedding(V + 1, d_model)   # 28 tokens → 64-dim
        self.pos_emb = nn.Embedding(L, d_model)        # 14 positions → 64-dim
        self.t_emb   = nn.Embedding(T + 1, d_model)   # 21 timesteps → 64-dim

        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=128,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head    = nn.Linear(d_model, V)  # output: logits over 27 clean tokens

    def forward(self, xt, t):
        pos = torch.arange(L, device=xt.device).unsqueeze(0)
        h = (
            self.tok_emb(xt)              # what tokens are currently visible?
            + self.pos_emb(pos)           # where in the sequence?
            + self.t_emb(t).unsqueeze(1)  # how much noise was applied?
        )
        return self.head(self.encoder(h))  # (B, L, V)
```

Four architectural choices that differ from a standard language model:

- **Bidirectional attention.** No causal mask. Every token attends to every other token. Context flows freely in both directions.
- **Timestep embedding.** A learned vector for each noise level `t`. At `t=1` the model should be very confident; at `t=19` it is constructively uncertain. Without this signal, it can't calibrate.
- **Non-autoregressive output.** All `L` positions are predicted in a single forward pass. There is no sequential dependency between outputs.
- **Loss on all positions.** Training minimizes cross-entropy at every position, not just masked ones. Visible tokens are easy; they ground the gradients for the hard masked ones.

---

## 4. Training: three lines that say everything

```python
for step in range(steps):
    x0 = data[torch.randint(len(data), (batch,))]    # random clean sentences
    t  = torch.randint(1, T + 1, (batch,))           # random noise level per example
    xt = torch.stack([q_forward(x0[b], t[b].item()) for b in range(batch)])

    logits = model(xt, t)                             # (B, L, V)
    loss   = F.cross_entropy(logits.reshape(-1, V), x0.reshape(-1))

    opt.zero_grad(); loss.backward(); opt.step()
```

The training loop is four steps, repeated thousands of times:

1. Sample a random sentence and a random noise level `t` from `[1, T]`
2. Corrupt the sentence at level `t` using the forward process
3. Ask the model to predict the original tokens at every position
4. Minimize cross-entropy between predictions and ground truth

Crucially, `t` is sampled uniformly across all noise levels. The model trains equally on "almost clean text" (t=1) and "almost fully masked text" (t=19). This is what makes the reverse process work: the model learns to denoise from any noise level, not just one.

---

## 5. The reverse process: the 1/t reveal schedule

At inference time, we start from a fully-masked sequence and iteratively reveal tokens over `T` steps. The clever part is the reveal rate: at step `t`, we reveal a fraction `1/t` of the still-masked positions.

<iframe src="/playgrounds/text-diffusion/fig3-reverse-process.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="Figure 3: reverse process and reveal schedule"></iframe>
<p class="fig-note">Left: the expected mask fraction tracks (t-1)/T, the exact inverse of the forward schedule. Right: click <strong>Step</strong> to walk through a generation. Green tokens are newly revealed at each step; they become context anchors for subsequent steps.</p>

```python
@torch.no_grad()
def sample(model, n=4):
    x = torch.full((n, L), MASK_ID, dtype=torch.long)  # all [MASK]

    for t in range(T, 0, -1):
        t_batch = torch.full((n,), t, dtype=torch.long)
        logits  = model(x, t_batch)
        probs   = F.softmax(logits, dim=-1)            # (n, L, V)

        sampled = torch.multinomial(probs.reshape(-1, V), 1).reshape(n, L)

        is_masked     = (x == MASK_ID)
        should_reveal = is_masked & (torch.rand_like(x, dtype=float) < 1.0 / t)
        x = torch.where(should_reveal, sampled, x)

    return [decode(x[j].tolist()) for j in range(n)]
```

Why `1/t` works: at `t=T` (the first step), reveal 1/20 = 5%, just one or two high-confidence tokens. At `t=10`, reveal 1/10 = 10% of what remains. At `t=1` (the final step), reveal 100%, filling everything still masked. The expected mask fraction after step `t` is exactly `(t-1)/T`, tracking the forward process in reverse.

Once a token is revealed, it stays. This creates a natural curriculum: high-confidence tokens are revealed early and become context anchors. The harder, ambiguous positions get filled later, using all those anchors as evidence.

---

## 6. Key observations from running the script

Running the toy script on its eight-sentence corpus surfaces cleanly instructive behaviors.

**It memorizes the corpus, intentionally.** With only eight training sentences and 4000 steps, the model converges to near-perfect reconstruction. This is the point: we want to confirm that the reverse process faithfully recovers known strings. Memorization here is signal, not overfit.

**Short common tokens anchor first.** Watching the generation trace, you reliably see spaces and common short words revealed in the first few denoising steps. The model is most confident about these because they appear in almost every training sentence and are unambiguous in context.

**The timestep embedding does real work.** Ablating the timestep embedding (removing `self.t_emb`) causes slower convergence and noisier generated sequences. The model genuinely uses the noise level to calibrate its output distributions. At `t=1`, it should output nearly one-hot distributions; at `t=19`, broader distributions encoding genuine uncertainty.

**Loss on all positions matters.** The training objective doesn't mask the loss for visible tokens: it computes cross-entropy at every position. The visible tokens are easy and converge quickly, but that easy signal provides clean gradients that help the harder masked positions learn faster. Masking the loss to only masked positions makes training noticeably less stable.

---

## 7. How the paradigms compare

<table class="comparison-table">
<thead>
<tr><th>Dimension</th><th>Autoregressive (LLM)</th><th>Text diffusion</th></tr>
</thead>
<tbody>
<tr><td>"Noise"</td><td>N/A</td><td>Random token masking</td></tr>
<tr><td>Generation order</td><td>Left-to-right, fixed</td><td>Confidence-first, global</td></tr>
<tr><td>Attention</td><td>Causal (looks left only)</td><td>Bidirectional (global context)</td></tr>
<tr><td>Revision</td><td>None; tokens are final</td><td>Later steps use global context</td></tr>
<tr><td>Parallelism</td><td>Inherently sequential</td><td>All positions updated per step</td></tr>
<tr><td>Training signal</td><td>Next-token prediction</td><td>Clean-token prediction at any noise level</td></tr>
<tr><td>Inference steps</td><td>One per token (linear in length)</td><td>Fixed T steps (independent of length)</td></tr>
<tr><td>Real-world examples</td><td>GPT-4, Claude, Llama</td><td>MDLM, SEDD, Gemma Diffusion</td></tr>
</tbody>
</table>

This isn't a winner vs. loser comparison. It's a different set of tradeoffs. Autoregressive models benefit from decades of optimization, fast KV-cache inference, and an extremely stable training objective. Text diffusion offers bidirectional context at generation time, a natural refinement loop, and inference time that doesn't scale with sequence length.

---

## 8. From 200 lines to production systems

This toy script is a conceptual skeleton. Real text diffusion models (MDLM, SEDD, Gemma Diffusion) build on exactly the same three ideas but add the engineering needed for quality at scale:

- **Much larger vocabularies.** Real systems use 50k+ BPE tokens rather than 27 characters. The masking logic is identical; the vocabulary just grows.
- **More principled noise beyond masking.** Absorbing-state diffusion and uniform noise (replacing masked tokens with random tokens rather than a fixed mask) can improve training signal.
- **Learned or cosine noise schedules.** The linear schedule here is the simplest option. Cosine schedules spend more training time in the intermediate noise levels where the learning signal is richest.
- **Analytic unmasking probabilities.** Rather than sampling from the model's softmax and using a Bernoulli reveal gate, production systems derive exact posterior probabilities for each token being its clean value given the noisy input.
- **Classifier-free guidance.** The same conditioning mechanism used in image diffusion (running the model twice, once conditioned and once not, then interpolating the outputs) applies directly to text diffusion for controlled generation.

The loop in `train()` and the reveal logic in `sample()` are directly recognizable in MDLM's codebase. The abstraction gap between this toy and a state-of-the-art system is mostly engineering, not conceptual, which is exactly why building it from scratch is so instructive.

> The most important thing the script demonstrates is that generation doesn't have to be directional. Text has structure in all directions simultaneously, and a model that sees and refines the full sequence at every step may be better positioned to exploit that structure than one that can only look left.

---

## The full script

```python
"""
toy_text_diffusion.py
Minimal text diffusion model — three core ideas:
  1. Forward process  — gradually corrupt text by masking tokens
  2. Training         — learn to predict clean tokens from masked ones
  3. Reverse process  — start fully masked, iteratively reveal tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS   = list("abcdefghijklmnopqrstuvwxyz ")
V       = len(CHARS)   # 27 clean token types
MASK_ID = V            # index 27
c2i     = {c: i for i, c in enumerate(CHARS)}

def encode(s, length):
    s = s.ljust(length)[:length]
    return [c2i.get(c, c2i[" "]) for c in s]

def decode(ids):
    return "".join(CHARS[i] if i < V else "█" for i in ids)

SENTENCES = [
    "hello world ", "cats and dogs", "sun sets slow ",
    "birds fly high", "rain hits hard ", "cold wind blows",
    "deep blue sky  ", "long dark night",
]
L    = max(len(s) for s in SENTENCES)
data = torch.tensor([encode(s, L) for s in SENTENCES])

T = 20

def mask_rate(t): return t / T

def q_forward(x0, t):
    corrupted = x0.clone()
    corrupted[torch.rand(x0.shape) < mask_rate(t)] = MASK_ID
    return corrupted

class DenoisingTransformer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.tok_emb = nn.Embedding(V + 1, d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        self.t_emb   = nn.Embedding(T + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head    = nn.Linear(d_model, V)

    def forward(self, xt, t):
        pos = torch.arange(L, device=xt.device).unsqueeze(0)
        h = self.tok_emb(xt) + self.pos_emb(pos) + self.t_emb(t).unsqueeze(1)
        return self.head(self.encoder(h))

def train(steps=3000, batch=32):
    model = DenoisingTransformer()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        idx = torch.randint(len(data), (batch,))
        x0  = data[idx]
        t   = torch.randint(1, T + 1, (batch,))
        xt  = torch.stack([q_forward(x0[b], t[b].item()) for b in range(batch)])
        loss = F.cross_entropy(model(xt, t).reshape(-1, V), x0.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")
    return model

@torch.no_grad()
def sample(model, n=4):
    x = torch.full((n, L), MASK_ID, dtype=torch.long)
    for t in range(T, 0, -1):
        t_batch = torch.full((n,), t, dtype=torch.long)
        probs   = F.softmax(model(x, t_batch), dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, V), 1).reshape(n, L)
        is_masked     = (x == MASK_ID)
        should_reveal = is_masked & (torch.rand_like(x, dtype=torch.float) < 1.0 / t)
        x = torch.where(should_reveal, sampled, x)
    return [decode(x[j].tolist()) for j in range(n)]

if __name__ == "__main__":
    model = train(steps=4000)
    print(sample(model, n=4))
```

---

*Further reading: [MDLM: Simplified and Improved Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.07524) · [SEDD: Score Entropy Discrete Diffusion](https://arxiv.org/abs/2310.16834) · [Gemma Diffusion](https://arxiv.org/abs/2506.07539)*
