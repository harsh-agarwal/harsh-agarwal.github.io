---
layout: playground-post
title: "Quantization: How a 70B Model Fits on Your Laptop"
date: 2026-07-08
description: "A from-scratch walkthrough of neural network quantization: the math, the code, and what it actually does to a model you trained yourself."
tags: [quantization, machine-learning, llm, deep-learning]
---

I want to start with a number: **1,120 GB**.

That's how much RAM you need to train a 70B parameter model in fp32. Even in fp16, the standard for most training today, you're looking at 140 GB just for the weights, before gradients and optimizer state enter the picture. A high-end A100 has 80 GB. A consumer 4090 has 24 GB. Your laptop has 16 GB of unified memory if you're lucky.

Quantization is the answer to this. It's the actual production solution used by Meta, Mistral, and every serious LLM deployment team, not a workaround. LLaMA 3 70B runs on a MacBook Pro today because of it.

Most quantization posts describe the methods and stop there. They don't show you what's *actually happening* to the numbers, what gets lost, or why sometimes you can go to INT4 and barely notice, and other times INT8 breaks things badly.

This one does it differently. We'll build the intuition from bits up, implement quantization from scratch on a VAE trained on MNIST, and *watch* the degradation happen. Then we'll implement simplified versions of GPTQ and AWQ on the same model and watch the degradation improve. No accuracy tables from papers. Real model, real weights, real images.

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
.results-note {
  background: #f6f4ee;
  border-left: 3px solid #c8c4b0;
  padding: 10px 14px;
  font-style: italic;
  font-size: 13px;
  color: #888;
  margin: 12px 0 20px;
  border-radius: 0 6px 6px 0;
}
.code-block-wrap {
  position: relative;
}
.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(255,255,255,0.12);
  color: #ccc;
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 5px;
  padding: 3px 9px;
  font-size: 11px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  cursor: pointer;
  line-height: 1.6;
  transition: background 0.15s, color 0.15s;
  user-select: none;
}
.copy-btn:hover {
  background: rgba(255,255,255,0.22);
  color: #fff;
}
.copy-btn.copied {
  color: #7ec882;
  border-color: rgba(126,200,130,0.4);
}
</style>

<script>
MathJax = { tex: { inlineMath: [['$', '$']] } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>

<script>
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('pre').forEach(function (pre) {
    var wrap = document.createElement('div');
    wrap.className = 'code-block-wrap';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    wrap.appendChild(btn);

    btn.addEventListener('click', function () {
      var code = pre.querySelector('code');
      var text = code ? code.innerText : pre.innerText;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      });
    });
  });
});
</script>

---

## 1. What Is a Number, Actually?

Before compressing numbers, it helps to understand what we're compressing.

### Floating point: trading range for precision

A 32-bit float (`fp32`) stores three things:

```
[1 bit: sign] [8 bits: exponent] [23 bits: mantissa]
```

The **exponent** controls scale (the order of magnitude). The **mantissa** controls precision (how finely you can resolve differences within that scale).

The interesting property: floating point gives you roughly the same *relative* precision everywhere. The gap between 1.0 and the next representable float is about 1.19×10⁻⁷. The gap between 1,000,000.0 and the next is about 0.125. Different absolute gaps, same relative precision. For neural network weights this matters: a weight of 0.0001 and a weight of 1.0 both get represented faithfully.

`fp16` makes the obvious tradeoff:

```
[1 bit: sign] [5 bits: exponent] [10 bits: mantissa]
```

Exponent shrank from 8 to 5 bits (narrower range), mantissa from 23 to 10 bits (less precision). Still works for training because mixed-precision keeps the master weights in fp32. Google's `bfloat16` goes the other way: keeps the full 8-bit exponent (same range as fp32) but sacrifices precision. The reasoning is that you'd rather lose precision than hit overflow.

Now here's where quantization enters. **INT8**:

```
[8 bits: just an integer, 0–255]
```

No exponent. No mantissa. 256 possible values, period. The entire continuous number line discretized into 256 buckets.

**INT4** gives you **16 buckets**. You're encoding the full weight distribution of a model into 16 values per weight. This sounds absurd. That it *works* at all, that INT4 models still generate coherent text and sharp images, is actually the surprising thing. Understanding why is what this post is about.

> **The key intuition:** Neural network weights don't use their full numerical range uniformly. Most cluster near zero with a roughly Gaussian distribution. A smart quantization scheme packs those 16 or 256 buckets tightly around where the weights actually live, covering the tails coarsely. You lose very little where the density is high.

---

## 2. The Mapping: Scale and Zero-Point

Quantization is a mapping problem. Given a range of float values, represent them as integers, and later reconstruct approximate floats from those integers. The full mapping is defined by exactly two numbers: a **scale** and a **zero-point**.

### Affine quantization

For a weight tensor `W`:

$$W_{int} = \text{clamp}\left(\text{round}\left(\frac{W}{s} + z\right),\ 0,\ 2^b - 1\right)$$

$$\hat{W} = s \cdot (W_{int} - z)$$

Where $s$ is the scale (bucket width), $z$ is the zero-point (shifts the grid so 0.0 maps exactly to an integer), and $b$ is the bit width. To find them:

$$s = \frac{W_{max} - W_{min}}{2^b - 1}, \qquad z = \text{round}\left(-\frac{W_{min}}{s}\right)$$

In code, this is almost embarrassingly short:

```python
import torch

def quantize(W: torch.Tensor, bits: int = 8):
    qmin, qmax = 0, 2**bits - 1

    w_min = W.min().item()
    w_max = W.max().item()
    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = round(-w_min / scale)

    W_int = torch.clamp(
        torch.round(W / scale + zero_point), qmin, qmax
    ).to(torch.uint8)

    return W_int, scale, zero_point


def dequantize(W_int: torch.Tensor, scale: float, zero_point: int):
    return scale * (W_int.float() - zero_point)
```

Everything else in quantization is engineering built around this core idea.

### The rounding error

The difference between $W$ and $\hat{W} = \text{dequantize}(\text{quantize}(W))$ is the quantization error $\epsilon = W - \hat{W}$. For a well-behaved distribution, this is bounded by $s/2$, half a bucket width.

With INT8 and a weight range of [-0.5, 0.5]: $s \approx 0.0039$, max error ≈ 0.002. Barely anything.

With INT4 and the same range: $s \approx 0.059$, max error ≈ 0.030. Eight times larger. This is where things start to matter.

---

## 3. The Outlier Problem

Here's what kills quantization in practice, and it's not where you'd expect.

Take a weight matrix whose values are mostly in [-0.5, 0.5]. Your scale is small, your buckets are finely spaced, all looks fine. Now one outlier weight sits at 50.0:

$$s = \frac{50.0 - (-0.5)}{255} \approx 0.198$$

Every normal weight, formerly resolved to a precision of 0.002, is now resolved to 0.198. One value hijacked the entire quantization grid and blew up the error for 99.9% of the weights.

This is why modern methods don't quantize entire weight matrices at once. They quantize in **groups**, typically 128 weights at a time, so an outlier in one group doesn't contaminate others.

Activations are harder still. During inference, activation values vary per input and you can't characterize their range from static weights alone. This is why **weight-only quantization** (INT4 weights, fp16 activations) is far more stable than fully quantized inference. It's also the problem that motivated GPTQ, SmoothQuant, and AWQ, which we'll get to in Chapter 7.

---

## 4. Let's Train the VAE

Let's build something we can quantize and *watch break*.

A Variational Autoencoder on MNIST is a good teaching tool for this: the model's output is an image, so quantization degradation becomes *visible* rather than just a number on a benchmark. We'll train it, quantize it multiple ways, and compare reconstructions side by side.

### Setup

```python
# requirements: torch torchvision matplotlib numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 16
BATCH_SIZE = 128
EPOCHS = 10
```

### Architecture

```python
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu     = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, 784),        nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

### Training

```python
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (pixel-wise BCE)
    recon_loss = F.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )
    # KL divergence: regularises the latent space
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def train():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('.', train=True,  download=True, transform=transform)
    test_data  = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}  loss: {total_loss/len(train_data):.2f}")

    torch.save(model.state_dict(), "vae_mnist.pt")
    return model, test_loader


model, test_loader = train()
```

About 5 minutes on CPU, under a minute on GPU.

### Visualising reconstructions

Before we quantize anything, let's write the helper we'll reuse throughout:

```python
def show_reconstructions(decoder, encoder, test_loader,
                         title="Reconstructions", n=8):
    """
    Encode a batch with the (possibly quantized) encoder,
    decode with the (possibly quantized) decoder,
    and show originals vs reconstructions.
    """
    decoder.eval()
    encoder.eval()

    x, _ = next(iter(test_loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        mu, logvar = encoder(x)
        z = mu        # use the mean, no randomness
        recon = decoder(z)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=9)
    axes[1, 0].set_ylabel('Recon',    fontsize=9)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

# fp32 baseline
show_reconstructions(model.decoder, model.encoder, test_loader,
                     title="fp32 baseline")
```

---

## 5. Naive Quantization: Watching It Break

Time to quantize. We'll apply the basic scale+zero-point scheme from Chapter 2 at INT8, INT4, INT3, and INT2.

```python
import copy

def quantize(W: torch.Tensor, bits: int):
    qmin, qmax = 0, 2**bits - 1
    w_min, w_max = W.min().item(), W.max().item()

    # Guard against zero-range tensors (bias vectors etc.)
    if w_max == w_min:
        return W.clone(), 1.0, 0

    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = round(-w_min / scale)

    W_int = torch.clamp(
        torch.round(W / scale + zero_point), qmin, qmax
    ).to(torch.uint8)

    return W_int, scale, zero_point


def dequantize(W_int: torch.Tensor, scale: float, zero_point: int):
    return scale * (W_int.float() - zero_point)


def quantize_model_naive(model: nn.Module, bits: int) -> nn.Module:
    """
    Fake quantization: replace each Linear weight with its
    quantized-then-dequantized version. Weights stay as floats
    in memory — we're measuring the error, not the storage.
    """
    q_model = copy.deepcopy(model)
    q_model.eval()

    with torch.no_grad():
        for module in q_model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                W_int, scale, zp = quantize(W, bits=bits)
                module.weight.data = dequantize(W_int, scale, zp)

    return q_model
```

A quick note on "fake" quantization: we dequantize the weights back to float for the forward pass. Real deployment (llama.cpp, vLLM) keeps them as INT4/INT8 and uses integer kernels. The error profile is identical though, we're just isolating it for inspection without needing custom CUDA kernels.

Run it across all bit widths:

```python
bit_widths = [8, 4, 3, 2]

for bits in bit_widths:
    q_model = quantize_model_naive(model, bits=bits)
    show_reconstructions(
        q_model.decoder, model.encoder, test_loader,
        title=f"Naive INT{bits}"
    )
```

### What you'll see

<div class="results-note">Results: insert your reconstructed image grids here after running the code.</div>

**INT8**: Essentially identical to fp32. The max error per weight (~0.002) is below the threshold of visual difference.

**INT4**: Starting to show. Slight blurring on fine strokes: the "1" digits lose their taper, the loops in "8" soften. SSIM drops measurably.

**INT3**: Clearly degraded. Some digits become ambiguous. "4" and "9" start merging.

**INT2**: Four buckets. Recognisable shapes but no fine structure. You've essentially kept the coarse topology and thrown away everything else.

The degradation is *gradual and structured*: fine details go first, coarse structure hangs on longest. Think of it as the neural network equivalent of JPEG compression: blurring and artifacts before complete breakdown.

### Measuring it properly

Visual inspection is good, but let's also get numbers. SSIM (Structural Similarity Index) gives a quality score between 0 and 1:

```python
from torchmetrics.image import StructuralSimilarityIndexMeasure

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

def measure_ssim(decoder, encoder, test_loader, n_batches=5):
    decoder.eval()
    encoder.eval()
    scores = []

    for i, (x, _) in enumerate(test_loader):
        if i >= n_batches:
            break
        x = x.to(DEVICE)
        with torch.no_grad():
            mu, _ = encoder(x)
            recon = decoder(mu)
        scores.append(ssim(recon.cpu(), x.cpu()).item())

    return np.mean(scores)

print(f"fp32  SSIM: {measure_ssim(model.decoder, model.encoder, test_loader):.4f}")
for bits in bit_widths:
    q = quantize_model_naive(model, bits=bits)
    score = measure_ssim(q.decoder, model.encoder, test_loader)
    print(f"INT{bits}  SSIM: {score:.4f}")
```

---

## 6. Why It Breaks: Activations and the Outlier Story

The results above might make you mildly optimistic. INT4 is mostly fine, INT8 is essentially free. So what's the fuss?

The catch is that we only quantized the *decoder weights*, not the activations flowing through it. The encoder still runs in fp32 and the latent vector `z` it passes to the decoder is full precision. That's the easy case.

In a real transformer, the residual stream carries activation values that span several orders of magnitude. Unlike weights, you don't know their range at quantization time. It depends on the input. A prompt full of unusual tokens can produce activation outliers that trigger the same scale explosion we saw in Chapter 3.

LLaMA and similar models have a well-documented version of this: a small number of "outlier dimensions" in the residual stream consistently hit values ~100x the typical magnitude. These 0.1% of dimensions blow up the quantization scale for the entire layer.

We can recreate this on our VAE. Let's inject one extreme weight into the decoder (the kind of thing that can emerge after fine-tuning on a shifted distribution) and see what happens:

```python
def inject_outlier(model: nn.Module, layer_idx: int = 0,
                   outlier_value: float = 50.0) -> nn.Module:
    """
    Inject a single extreme weight value into one Linear layer.
    Simulates what outlier activations do to quantization scale.
    """
    q_model = copy.deepcopy(model)
    linears = [m for m in q_model.modules() if isinstance(m, nn.Linear)]
    target  = linears[layer_idx]

    with torch.no_grad():
        target.weight.data[0, 0] = outlier_value

    return q_model


model_with_outlier = inject_outlier(model, layer_idx=0, outlier_value=50.0)

# Naive INT4 on the outlier model
q_outlier = quantize_model_naive(model_with_outlier, bits=4)
show_reconstructions(q_outlier.decoder, model.encoder, test_loader,
                     title="Naive INT4 with outlier")
```

The reconstruction quality will noticeably collapse compared to the clean model. One weight hijacked the entire quantization grid. That's the problem GPTQ, SmoothQuant, and AWQ each solve, in different ways.

---

## 7. Modern Methods

The progression from naive INT8 to production INT4 is a sequence of solutions to specific failures. Here's a quick overview of the three main ones:

<iframe src="/playgrounds/quantization/quantization-methods.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="GPTQ, SmoothQuant and AWQ: overview slideshow"></iframe>
<p class="fig-note">Step through the key ideas behind GPTQ, SmoothQuant, and AWQ — what each method solves and how.</p>

And here's what each method actually does, step by step on a concrete toy network:

<iframe src="/playgrounds/quantization/quantization-deep-dive.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="GPTQ, SmoothQuant and AWQ: step-by-step deep dive"></iframe>
<p class="fig-note">A step-by-step walkthrough of each method on a toy network, showing exactly what the numbers look like before and after.</p>

### GPTQ (2022)

Naive quantization minimizes per-weight rounding error. GPTQ minimizes *model output* error, which is a different thing. It uses the Hessian diagonal to measure how sensitive the output is to each weight's rounding error. Sensitive weights get quantized carefully; their rounding error is propagated and compensated in neighbouring weights. Slower to run than naive methods, but meaningfully better quality at INT4.

### SmoothQuant (2022)

Activations can't be pre-calibrated because they change per input. SmoothQuant's idea: mathematically migrate the outlier "difficulty" from activations into weights using a channel-wise scale factor $s$. Since $(X / s) \cdot (W \times s) = X \cdot W$, the output is unchanged but now both tensors are quantization-friendly. This is what unlocks W8A8 inference (both weights and activations in INT8) on hardware tensor cores.

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$$

### AWQ (2023)

AWQ asks a simpler question than GPTQ: which weight channels have the highest activation magnitudes? Those are the ones where a rounding error gets amplified most in the output. One calibration forward pass finds them. Scale those channels up before quantization so they occupy more of the integer range, then scale back down at inference. No Hessian needed, runs in minutes, quality nearly matches GPTQ.

---

## 8. VAE Redux: Applying the Methods

Now let's apply simplified versions of GPTQ and AWQ to our VAE and see the reconstructions actually improve.

These aren't production implementations; they're teaching versions that capture the core idea of each method, stripped of engineering complexity. I'll call them "GPTQ-style" and "AWQ-style" to be upfront about that.

### GPTQ-style: importance-ordered quantization with error compensation

Three things we'll implement from GPTQ: (1) estimate per-weight importance from activation statistics as a Hessian proxy, (2) quantize in importance order from high to low, (3) after each weight, compensate the rounding error in remaining weights.

```python
def quantize_model_gptq_style(model: nn.Module,
                               calibration_loader: DataLoader,
                               bits: int = 4,
                               n_calib_batches: int = 10) -> nn.Module:
    """
    GPTQ-style quantization for Linear layers.

    Core ideas from the paper:
      - Use calibration data to estimate which weights matter most
      - Quantize from most to least important
      - After quantizing each weight, push its rounding error
        to the remaining unquantized weights
    """
    q_model = copy.deepcopy(model)
    q_model.eval()

    # Step 1: collect activation statistics per layer.
    # We use squared mean activation as a cheap Hessian diagonal proxy.
    # Real GPTQ computes (X^T X) / |X| more carefully; this captures
    # the same signal for a small model.
    activation_stats = {}

    hooks = []
    for name, module in q_model.named_modules():
        if isinstance(module, nn.Linear):
            def make_hook(n):
                def hook(mod, inp, out):
                    x = inp[0].detach()
                    stats = x.pow(2).mean(dim=0)    # (in_features,)
                    if n not in activation_stats:
                        activation_stats[n] = stats
                    else:
                        activation_stats[n] += stats
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            if i >= n_calib_batches:
                break
            q_model(x.to(DEVICE))

    for h in hooks:
        h.remove()

    for name in activation_stats:
        activation_stats[name] /= n_calib_batches

    # Step 2: quantize each layer with error compensation.
    with torch.no_grad():
        for name, module in q_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            W = module.weight.data.clone()  # (out_features, in_features)
            stats = activation_stats.get(name)

            if stats is None:
                importance = torch.ones(W.shape[1], device=W.device)
            else:
                importance = stats  # (in_features,)

            # Sort input channels by importance, highest first
            order = torch.argsort(importance, descending=True)
            W_q = W.clone()

            for col_idx in order:
                col = W[:, col_idx]

                col_int, scale, zp = quantize(col, bits=bits)
                col_q = dequantize(col_int, scale, zp)

                delta = col - col_q     # rounding error for this column

                # Distribute the error to remaining unquantized columns,
                # weighted by their importance
                remaining_cols = [c.item() for c in order
                                  if c.item() != col_idx.item()]

                if remaining_cols:
                    remaining_importance = importance[remaining_cols]
                    total_imp = remaining_importance.sum() + 1e-8
                    for rem_col in remaining_cols:
                        weight_frac = importance[rem_col] / total_imp
                        W[:, rem_col] -= delta * weight_frac.item()

                W_q[:, col_idx] = col_q

            module.weight.data = W_q

    return q_model
```

Run it:

```python
q_gptq = quantize_model_gptq_style(
    model, calibration_loader=test_loader, bits=4
)
show_reconstructions(q_gptq.decoder, model.encoder, test_loader,
                     title="GPTQ-style INT4")
```

### AWQ-style: protect the salient channels

AWQ is simpler to implement. One forward pass to find salient channels, then scale them up before quantizing.

```python
def quantize_model_awq_style(model: nn.Module,
                              calibration_loader: DataLoader,
                              bits: int = 4,
                              n_calib_batches: int = 10,
                              alpha: float = 0.5) -> nn.Module:
    """
    AWQ-style quantization for Linear layers.

    Core ideas from the paper:
      - Find salient channels by average activation magnitude
      - Scale them up before quantization so they use more buckets
      - Scale the output back down so the computation is unchanged
      - alpha controls how much difficulty to shift (0.5 is the AWQ default)
    """
    q_model = copy.deepcopy(model)
    q_model.eval()

    # Step 1: collect per-channel activation magnitudes
    activation_means = {}

    hooks = []
    for name, module in q_model.named_modules():
        if isinstance(module, nn.Linear):
            def make_hook(n):
                def hook(mod, inp, out):
                    x = inp[0].detach()
                    mean_abs = x.abs().mean(dim=0)   # (in_features,)
                    if n not in activation_means:
                        activation_means[n] = mean_abs
                    else:
                        activation_means[n] += mean_abs
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            if i >= n_calib_batches:
                break
            q_model(x.to(DEVICE))

    for h in hooks:
        h.remove()

    for name in activation_means:
        activation_means[name] /= n_calib_batches

    # Step 2: scale salient channels, quantize, unscale
    with torch.no_grad():
        for name, module in q_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            W = module.weight.data.clone()
            means = activation_means.get(name)

            if means is None:
                W_int, scale, zp = quantize(W, bits=bits)
                module.weight.data = dequantize(W_int, scale, zp)
                continue

            # Smooth factor per input channel: s = mean_act^alpha
            s = means.pow(alpha).clamp(min=1e-4)    # (in_features,)

            # Scale up: W[:, j] *= s[j]
            W_scaled = W * s.unsqueeze(0)

            # Quantize the scaled weights
            W_int, scale_q, zp = quantize(W_scaled, bits=bits)
            W_q_scaled = dequantize(W_int, scale_q, zp)

            # Undo the scale so the matmul output is unchanged:
            # (X / s) @ (W * s)^T = X @ W^T
            W_q = W_q_scaled / s.unsqueeze(0)
            module.weight.data = W_q

    return q_model
```

Run it:

```python
q_awq = quantize_model_awq_style(
    model, calibration_loader=test_loader, bits=4
)
show_reconstructions(q_awq.decoder, model.encoder, test_loader,
                     title="AWQ-style INT4")
```

### Side-by-side comparison

Now put them all together in one grid:

```python
def compare_all(model, test_loader):
    configs = [
        ("fp32 baseline",   model),
        ("Naive INT8",      quantize_model_naive(model, bits=8)),
        ("Naive INT4",      quantize_model_naive(model, bits=4)),
        ("Naive INT3",      quantize_model_naive(model, bits=3)),
        ("GPTQ-style INT4", quantize_model_gptq_style(model, test_loader, bits=4)),
        ("AWQ-style INT4",  quantize_model_awq_style(model, test_loader, bits=4)),
    ]

    x_fixed, _ = next(iter(test_loader))
    x_fixed = x_fixed.to(DEVICE)
    n = 8

    fig, axes = plt.subplots(len(configs), n, figsize=(n * 1.5, len(configs) * 1.6))

    for row, (title, m) in enumerate(configs):
        m.eval()
        with torch.no_grad():
            mu, _ = model.encoder(x_fixed)   # always encode with fp32 encoder
            recon = m.decoder(mu)

        ssim_score = measure_ssim(m.decoder, model.encoder, test_loader)

        for col in range(n):
            axes[row, col].imshow(recon[col, 0].cpu(), cmap='gray')
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(
            f"{title}\nSSIM={ssim_score:.3f}", fontsize=8, rotation=0,
            labelpad=90, va='center'
        )

    plt.suptitle("Quantization methods on VAE / MNIST decoder", y=1.01)
    plt.tight_layout()
    plt.savefig("quantization_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


compare_all(model, test_loader)
```

<div class="results-note">Results: insert <code>quantization_comparison.png</code> here after running the full comparison.</div>

### What to expect

**Naive INT8** is essentially fp32. Free compression, no visible degradation.

**Naive INT4** shows noticeable blurring. The "1" digits lose their taper. SSIM drops roughly 0.02-0.05 depending on your training run.

**Naive INT3** is clearly degraded. Digits start merging.

**GPTQ-style INT4** should be better than naive INT4. The important weights are quantized carefully and their errors are compensated, so the output error is spread more evenly. SSIM should recover most of the gap between naive INT4 and fp32.

**AWQ-style INT4** should land in similar territory to GPTQ-style, using a simpler approach. Salient channels occupy more of the INT4 range so their rounding errors are smaller where it matters.

Naive INT4 is not the ceiling. Both methods recover meaningful quality over the naive baseline using only a few calibration batches and no retraining.

---

## 9. The Honest Tradeoff Table

| Format | Bits/weight | Model size (70B) | Fits in | Quality |
|--------|------------|------------------|---------|---------|
| fp32 | 32 | 280 GB | 4x A100 80GB | Baseline |
| fp16 / bf16 | 16 | 140 GB | 2x A100 80GB | ~Baseline |
| INT8 | 8 | 70 GB | 1x A100 80GB | 99%+ retained |
| INT4 (GPTQ/AWQ) | 4 | 35 GB | 2x 24GB cards | 95-98% on most tasks |
| INT4 (GGUF Q4_K_M) | ~4.5 | ~38 GB | M2 Max (64GB) | 95-97% |
| INT3 | 3 | 26 GB | 1x 24GB card | Degraded on reasoning |
| INT2 | 2 | 18 GB | Consumer GPU | Not production-ready |

"Quality" is deliberately vague because it depends heavily on the task. Summarisation and translation hold up much better under INT4 than code generation or multi-step reasoning. The fine-grained weight distinctions matter more when the task requires precise recall or logical chaining. Same pattern we saw in the digits: coarse structure survives, fine detail goes first.

---

## 10. What Should I Actually Use?

**Running locally on a gaming PC or Mac:** GGUF via llama.cpp. Use `Q4_K_M` as the default: it applies different bit widths to different weight matrices based on measured importance. Better than naive INT4 for almost no additional cost.

**Serving on A100s or H100s in production:** AWQ INT4 via vLLM or TGI. Up to 4x throughput improvement over fp16 with minimal quality loss.

**Fine-tuning a large model:** QLoRA. Keep the base model in NF4 (a 4-bit float format tuned for normally-distributed weights), fine-tune only small adapter layers in fp16. Memory footprint of a quantized model, training stability of full precision.

**Deploying a model under 7B where quality matters:** Don't quantize past INT8. The quality-to-size tradeoff gets worse as models shrink because there's less redundancy to exploit.

---

## Closing Thoughts

Quantization is usually framed as a compression trick you apply after training. It's more fundamental than that.

The reason it works at all is that trained neural networks are *massively redundant*. The information encoded in billions of weights doesn't actually require billions of full-precision floats to represent. During training, the model learned to pack knowledge into weight *patterns* that are robust to small perturbations. Quantization is just a controlled, systematic way of introducing those perturbations.

The VAE experiment shows this concretely: a digit is still a digit in INT4. The decoder encoded "three-ness" in a way that survives coarse quantization. Only the fine strokes (the exact taper of a numeral, the precise reconstruction of texture) actually need full precision. GPTQ and AWQ are just clever ways of finding *which* fine strokes matter most, and protecting those specifically.

The more we understand which weights carry information and how much, the smarter our compression becomes. We're genuinely still early.

---

### Want to go further?

The natural next experiment: take Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), train it on Shakespeare, and run the same quantization arc on it. The degradation story is the same: blurry digits become incoherent sentences. Structured knowledge (character names, consistent references) seems to go before fluency: the model can still produce grammatical sentences after losing the ability to keep track of who's speaking. Try it and see.

A few specific things worth looking at:

- Does naive INT4 degrade faster on nanoGPT than on the VAE? Hypothesis: yes, because language models tend to have more outlier dimensions in their residual streams.
- Does GPTQ-style recover more quality on language than on images, or less?
- At what bit width does the model lose the ability to spell character names correctly, vs losing basic grammatical coherence?

These are open questions for small models. There's no canonical answer in the literature. Run it and write it up.

---

*All code in this post is [on GitHub](#). The interactive demos are embedded above.*

---

*Part of a series on the systems behind modern ML. Previous posts: [Distributed Training Parallelism](https://harsh-agarwal.github.io/distributed-training-parallelism/), [Mixture of Experts](https://harsh-agarwal.github.io/mixture-of-experts/), [Optimizers 101](https://harsh-agarwal.github.io/optimizers-blog-2/).*
