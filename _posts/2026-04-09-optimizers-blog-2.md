---
layout: playground-post
title: "Optimizers 101: From SGD to Muon, and Why Toy Problems Lie to You"
date: 2026-04-09
playground_script: /playgrounds/muon-demo.js
---

I recently went down an optimizer rabbit hole. It started innocently — I wanted to build one of those classic Alec Radford-style contour visualizations where you watch SGD, Adam, and friends race toward a minimum. By the end, I'd built two completely different demos, read a dozen papers, and learned that the most important new optimizer in deep learning literally *cannot* show its advantage on the toy problems we've been using for a decade.

Here's what I found.

## The cast of characters

Let's start with the four optimizers I wanted to compare, and what each one actually does to a gradient before applying it.

### SGD with momentum

The OG. Compute the gradient, keep a running velocity, step in that direction.

$$v_t = \mu \cdot v_{t-1} + g_t$$

$$\theta_t = \theta_{t-1} - \alpha \cdot v_t$$

Momentum (typically μ=0.9) smooths out the noise and helps you barrel through flat regions. The problem: it treats every parameter identically. If the loss surface is a narrow valley (like Rosenbrock's banana), SGD bounces between the walls instead of sliding down the floor.

### Adam

Kingma & Ba (2014) combined momentum with per-parameter adaptive learning rates. The key insight: track both the first moment (mean of gradients) and second moment (mean of squared gradients), then divide one by the other.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(momentum)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(RMSprop-like)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \qquad \text{(bias correction)}$$

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

The division by √v̂ means parameters with consistently large gradients get smaller effective learning rates, and parameters with small gradients get larger ones. This is why Adam handles saddle points and ravines better than SGD — it automatically rescales per coordinate.

### AdamW

Loshchilov & Hutter (2017) noticed something subtle: the way Adam implements weight decay is mathematically wrong. Standard L2 regularization adds λθ to the gradient *before* the adaptive scaling, which means the regularization itself gets scaled down by the second moment. AdamW fixes this by applying weight decay *directly to the parameters*, decoupled from the adaptive step:

$$\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \cdot \theta_{t-1} \right)$$

Small change, significant impact. AdamW became the de facto standard for training transformers and has held that position for years.

### Muon

And then there's the new kid. Muon (MomentUm Orthogonalized by Newton-Schulz), introduced by Keller Jordan in 2024 during the NanoGPT speedrunning era, does something fundamentally different from all three above.

Where Adam adapts learning rates *per parameter* (element-wise), Muon treats the entire weight matrix as a geometric object. It:

1. Accumulates momentum (Nesterov-style)
2. Takes the SVD of the momentum buffer: B = UΣVᵀ
3. **Replaces all singular values with 1**: update = UVᵀ
4. Scales to match AdamW's update RMS

That step 3 is the magic. The orthogonalization preserves the *directions* the gradient points in but removes the *magnitude imbalance* between them. In practice, this means Muon explores the full parameter space uniformly instead of getting stuck hammering the same dominant directions.

## The classic visualization: contour plots

To build intuition for how these optimizers behave, I built an interactive contour visualization in the style of Alec Radford's famous animations from CS231n. You pick a 2D test function (Rosenbrock, Beale, Himmelblau, Ackley), and all four optimizers start from the same point and race toward the minimum.

<iframe src="/playgrounds/optimizer-viz.html" style="width:100%; height:860px; border:none; border-radius:8px; margin: 1.5em 0; display:block;" loading="lazy" title="Optimizer comparison visualization" ></iframe>

*The Radford-style contour plot with SGD (red), Adam (green), AdamW (blue), and Muon (amber) racing across loss surfaces. Gold stars mark the global optima. Toggle optimizers on/off, adjust individual learning rates, click to set custom start points.*

A few things jump out immediately:

- **SGD oscillates wildly** in narrow valleys (try Rosenbrock). The momentum keeps overshooting the walls.
- **Adam and AdamW navigate ravines smoothly** — the per-parameter scaling lets them take big steps along the valley floor and small steps perpendicular to it.
- **Himmelblau is fun** — it has four equal global minima, and different optimizers often converge to different ones depending on their trajectory.
- **Muon... is underwhelming.** It converges (or sometimes won't even converge), but it doesn't clearly beat Adam on any surface.

That last point bugged me. Muon is setting training speed records on NanoGPT, it's being used at trillion-parameter scale (Kimi K2), and papers show it's ~35% more token-efficient than AdamW. So why does it look mediocre on Beale's function?

## The epiphany: Muon can't win on 2D surfaces

After digging into the literature, the answer became clear: **Muon's entire advantage is structural, and it literally cannot manifest in 2D scalar optimization.**

Muon's core operation is Newton-Schulz orthogonalization of a matrix. It takes the gradient *matrix* G, computes its SVD (G = UΣVᵀ), and replaces it with UVᵀ — setting all singular values to 1. This rebalances the update so that every singular direction gets equal weight.

But for a 2D parameter vector (which is what our contour plots optimize), there's only **one** singular value. There's nothing to rebalance. The concept of "ill-conditioned gradient matrix" doesn't exist when your matrix is just a vector with two entries.

This is like testing a Formula 1 car's cornering advantage in a parking lot. The engineering that makes Muon fast — exploiting the matrix structure of weight matrices, removing spectral imbalance, treating parameters as coordinated transformations rather than independent scalars — needs *actual matrices* to operate on.

A key quote from Keller Jordan's blog: *"By design, Muon only applies to 2D parameters (and convolutional filters via flattening)."* It's a matrix optimizer being forced into a scalar world.

This made me realize: if we want to demonstrate Muon's advantage, we need a toy problem that's actually a *matrix optimization problem*.

## A better toy problem: associative memory

Recent work by Wang et al. (2025) gave exactly the right framing. They showed that Muon's advantage comes from how it handles *associative memory* — the key-value storage that happens in transformer attention layers and FFNs.

Here's the setup: an 8×8 weight matrix W must learn to store 8 key→value associations:

| Pair | Key | Value | Training frequency |
|------|-----|-------|--------------------|
| #1 | cat | animal | ~50% |
| #2 | dog | pet | ~16% |
| #3 | car | vehicle | ~8% |
| ... | ... | ... | ... |
| #8 | quokka | marsupial | <1% |

The matrix must satisfy W × key_i ≈ value_i for all pairs. This is exactly what attention layers do — they store facts as outer products in their weight matrices.

The key detail: training data follows a **Zipf distribution**. "Cat" shows up in almost every batch. "Quokka" barely ever appears. This mirrors real language data, where a few patterns are extremely common and a long tail of rare-but-important facts exists.

### Why Muon wins here

Each training batch produces a gradient matrix G. Because "cat" dominates, G's SVD looks something like:

$$G = U \times \operatorname{diag}(50,\ 12,\ 3,\ 0.8,\ \ldots) \times V^\top \qquad \leftarrow \text{cat direction dominates}$$

**SGD** follows this gradient directly — it over-optimizes "cat" and barely touches "quokka."

**Adam** normalizes each *element* independently by its running variance. This helps, but it's still working element-by-element. It doesn't understand that G is a matrix with structure. The singular spectrum of the learned weight matrix stays uneven.

**Muon** orthogonalizes the gradient:

$$\underbrace{G = U \times \operatorname{diag}(50,\ 12,\ 3,\ 0.8,\ \ldots) \times V^\top}_{\text{before: cat dominates}}$$

$$\underbrace{O = U \times \operatorname{diag}(1,\ 1,\ 1,\ 1,\ \ldots) \times V^\top}_{\text{after: all directions equal}}$$

Every direction — including the "quokka" direction — gets the same update magnitude. The directions themselves still come from the data (U and V are unchanged), so Muon isn't ignoring the gradient. It's just removing the magnitude imbalance.

The result: Muon learns rare facts nearly as well as common ones, while Adam and SGD systematically neglect the tail.

<div id="playground-muon-root" style="margin: 2em 0;"></div>

*Watch all four optimizers train an 8×8 weight matrix on Zipf-distributed key-value pairs. The per-pair error bars show how well each optimizer learned each fact — pay attention to the bottom rows (rare pairs). The SVD entropy metric quantifies how isotropic each optimizer's weight matrix is.*

Run this for ~500 steps and watch the per-pair error bars. The pattern is striking: Adam and SGD show low error for "cat" and "dog" but stubbornly high error for "sushi" and "quokka." Muon's bars shrink uniformly across all rows.

The SVD entropy metric at the bottom confirms the mechanism: Muon consistently produces a more isotropic weight matrix (higher entropy = energy distributed evenly across all singular directions), while Adam's weight matrix develops a skewed spectrum.

Try this experiment: crank up Adam's learning rate to match Muon's. You'll see the common pairs start oscillating while the rare pairs *still* lag behind. Adam's problem isn't step *size* — it's step *direction*.

## Final thoughts

**Toy problems are useful until they aren't.** Radford's contour visualizations remain excellent for building intuition about momentum, adaptive learning rates, and saddle point behavior. But the optimizer landscape has moved past what 2D surfaces can express. If your evaluation methodology can't distinguish between Adam and Muon, the methodology is wrong — not the optimizer. As the field moves toward matrix-aware and geometry-aware optimization, we need toy problems that preserve the structural properties these methods exploit.

**Muon and Adam aren't competing — they're complementary.** Adam provides element-wise adaptivity: per-parameter learning rates derived from gradient statistics. Muon provides spectral adaptivity: matrix-level rebalancing of update directions via orthogonalization. These address fundamentally different failure modes. Every production Muon deployment (Moonlight, Kimi K2, NanoGPT speedruns) uses a hybrid setup — Muon on hidden layer weight matrices, AdamW on embeddings, biases, and normalization layers. Treating them as substitutes misses the point.

**Data distribution is an optimizer design constraint.** Muon's advantage scales with the heavy-tailedness of the training data. Natural language is massively Zipfian — a handful of tokens and patterns dominate, while the long tail carries most of the knowledge. Orthogonalization directly counteracts the gradient imbalance this creates. For uniformly distributed data, the advantage shrinks. This isn't a limitation — it's a design match. The right optimizer depends on the spectral properties of your gradients, which are shaped by your data.

**Mechanisms beat benchmarks.** You can always tune a learning rate to make Adam look competitive on a specific task. But understanding that Muon works via spectral rebalancing of the gradient matrix tells you *when* to reach for it (heavy-tailed matrix optimization at scale), *when not to* (scalar parameters, 1D vectors, uniform data), and *what to monitor* (singular value entropy of your weight matrices). That kind of reasoning transfers across tasks. A benchmark number doesn't.

## Further reading

- [Keller Jordan's Muon blog post](https://kellerjordan.github.io/posts/muon/) — the original design document
- [Jeremy Bernstein's derivation](https://jeremybernste.in/writing/deriving-muon) — the theoretical foundations
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — Moonshot's scaling paper
- [Muon Outperforms Adam in Tail-End Associative Memory Learning](https://arxiv.org/abs/2509.26030) — the paper that explains *why* Muon wins
- [Sebastian Ruder's overview of gradient descent optimization](https://www.ruder.io/optimizing-gradient-descent/) — still the best general reference
- [CS231n optimization notes](https://cs231n.github.io/neural-networks-3/) — where Radford's animations live

---

*The interactive demos in this post are built with vanilla JavaScript (contour plot) and React (matrix memory). The contour visualization uses marching squares for contour lines and implements all four optimizers from scratch. The associative memory demo uses power-iteration SVD for the Newton-Schulz orthogonalization. Both run entirely in the browser with no dependencies.*
