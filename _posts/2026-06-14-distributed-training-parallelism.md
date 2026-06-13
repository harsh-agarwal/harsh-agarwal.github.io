---
layout: playground-post
title: "A Practitioner's Guide to Distributed Training Parallelism"
date: 2026-06-14
playground_script: /playgrounds/parallelism-demo.js
tags: [distributed-training, parallelism, deep-learning, gpu, infrastructure]
description: "DDP, FSDP, Pipeline, Tensor, and 3D parallelism — what each actually does, when each breaks, and how to choose."
---

Training a large model isn't hard because the math is complicated. It's hard because the model doesn't fit. A 70-billion-parameter transformer needs roughly 140 GB just to store its fp16 weights — and training requires 4× that for gradients and optimizer states. An 80 GB GPU can't hold it. So you split the work.

But *how* you split it determines whether your training run is efficient or whether half your GPUs are sitting idle waiting for each other. This post covers the five main parallelism strategies, not as textbook definitions but as a field guide: what each one actually does to your GPUs, where the hidden costs are, and when to use which.

---

<style>
.fig{background:#fff;border:1px solid #DDE3ED;border-radius:12px;padding:24px;margin:28px 0;box-shadow:0 1px 4px rgba(30,40,70,0.05);}
.fig h3{font-family:'SF Mono','Cascadia Code',Consolas,monospace;font-size:12px;font-weight:600;color:#6B7A94;margin-bottom:12px;letter-spacing:0.04em;}
.fig svg text{font-family:'SF Mono','Cascadia Code',Consolas,monospace;}
.fig figcaption{font-size:12px;color:#6B7A94;margin-top:10px;line-height:1.5;}
#parallelism-root{margin:40px 0;border-radius:16px;overflow:hidden;border:1px solid #DDE3ED;box-shadow:0 2px 12px rgba(30,40,70,0.07);}
.demo-loading{text-align:center;padding:60px 20px;color:#6B7A94;font-family:'SF Mono',monospace;font-size:13px;background:#F4F6FB;border-radius:16px;}
</style>

## 1. Distributed Data Parallel (DDP)

DDP is the simplest strategy and where almost everyone should start. Every GPU gets a complete copy of the model. The training data is split so each GPU processes a different mini-batch. After the backward pass, gradients are averaged across all GPUs via an AllReduce, then every GPU applies the same optimizer step with the same averaged gradients — keeping the replicas in sync.

<figure class="fig" id="fig-ddp">
<h3>Figure 1 — Distributed Data Parallel (DDP)</h3>
<svg viewBox="0 0 460 210" width="100%">
  <rect x="20" y="8" width="90" height="24" rx="4" fill="#CFDDFA" stroke="#4E6AD4" stroke-width="1"/>
  <text x="65" y="24" text-anchor="middle" fill="#4E6AD4" font-size="9" font-weight="600">Batch 0</text>
  <rect x="135" y="8" width="90" height="24" rx="4" fill="#FADDC8" stroke="#C4822A" stroke-width="1"/>
  <text x="180" y="24" text-anchor="middle" fill="#C4822A" font-size="9" font-weight="600">Batch 1</text>
  <rect x="250" y="8" width="90" height="24" rx="4" fill="#DACBFA" stroke="#7E5AC0" stroke-width="1"/>
  <text x="295" y="24" text-anchor="middle" fill="#7E5AC0" font-size="9" font-weight="600">Batch 2</text>
  <rect x="365" y="8" width="90" height="24" rx="4" fill="#C5ECD6" stroke="#24956E" stroke-width="1"/>
  <text x="410" y="24" text-anchor="middle" fill="#24956E" font-size="9" font-weight="600">Batch 3</text>
  <line x1="65" y1="34" x2="65" y2="50" stroke="#4E6AD4" stroke-width="1.2"/>
  <line x1="180" y1="34" x2="180" y2="50" stroke="#C4822A" stroke-width="1.2"/>
  <line x1="295" y1="34" x2="295" y2="50" stroke="#7E5AC0" stroke-width="1.2"/>
  <line x1="410" y1="34" x2="410" y2="50" stroke="#24956E" stroke-width="1.2"/>
  <rect x="20" y="52" width="90" height="50" rx="6" fill="#fff" stroke="#4E6AD4" stroke-width="1.3"/>
  <text x="65" y="72" text-anchor="middle" fill="#4E6AD4" font-size="10" font-weight="600">GPU 0</text>
  <text x="65" y="90" text-anchor="middle" fill="#6B7A94" font-size="8">Full Model</text>
  <rect x="135" y="52" width="90" height="50" rx="6" fill="#fff" stroke="#4E6AD4" stroke-width="1.3"/>
  <text x="180" y="72" text-anchor="middle" fill="#4E6AD4" font-size="10" font-weight="600">GPU 1</text>
  <text x="180" y="90" text-anchor="middle" fill="#6B7A94" font-size="8">Full Model</text>
  <rect x="250" y="52" width="90" height="50" rx="6" fill="#fff" stroke="#4E6AD4" stroke-width="1.3"/>
  <text x="295" y="72" text-anchor="middle" fill="#4E6AD4" font-size="10" font-weight="600">GPU 2</text>
  <text x="295" y="90" text-anchor="middle" fill="#6B7A94" font-size="8">Full Model</text>
  <rect x="365" y="52" width="90" height="50" rx="6" fill="#fff" stroke="#4E6AD4" stroke-width="1.3"/>
  <text x="410" y="72" text-anchor="middle" fill="#4E6AD4" font-size="10" font-weight="600">GPU 3</text>
  <text x="410" y="90" text-anchor="middle" fill="#6B7A94" font-size="8">Full Model</text>
  <line x1="112" y1="77" x2="133" y2="77" stroke="#4E6AD4" stroke-width="1" stroke-dasharray="4,3"/>
  <polygon points="133,77 127,74 127,80" fill="#4E6AD4"/>
  <line x1="227" y1="77" x2="248" y2="77" stroke="#4E6AD4" stroke-width="1" stroke-dasharray="4,3"/>
  <polygon points="248,77 242,74 242,80" fill="#4E6AD4"/>
  <line x1="342" y1="77" x2="363" y2="77" stroke="#4E6AD4" stroke-width="1" stroke-dasharray="4,3"/>
  <polygon points="363,77 357,74 357,80" fill="#4E6AD4"/>
  <path d="M65,104 Q8,148 230,152 Q452,148 410,104" fill="none" stroke="#4E6AD4" stroke-width="1.2" stroke-dasharray="5,3" opacity="0.5"/>
  <rect x="156" y="138" width="148" height="24" rx="5" fill="#EDF0FA" stroke="#4E6AD4" stroke-width="1"/>
  <text x="230" y="154" text-anchor="middle" fill="#4E6AD4" font-size="10" font-weight="600">AllReduce ∇ gradients</text>
  <text x="230" y="188" text-anchor="middle" fill="#6B7A94" font-size="8">Same weights → different data → averaged gradients</text>
</svg>
<figcaption>DDP replicates the full model on every GPU. Each GPU processes a different data shard. Gradients are averaged via a ring-AllReduce after the backward pass.</figcaption>
</figure>

### Practical nuances

**Why it's fast.** PyTorch DDP overlaps the gradient AllReduce with the backward pass. The moment a layer's gradient is ready, it's bucketed and communication starts — while the next layer is still computing. This overlap means the communication cost is largely hidden behind computation for reasonably-sized models.

**Where it breaks.** DDP doesn't reduce memory at all. Every GPU stores the full model parameters, full gradients, and full optimizer states. For a 7B model with mixed-precision Adam, that's 6.7B × 16 bytes = ~107 GB per GPU — already over the 80 GB limit of an A100/H100. The moment the model exceeds single-GPU memory, DDP is out.

**The scaling ceiling.** DDP throughput scales nearly linearly up to 256–512 GPUs for large batch sizes. Beyond that, the AllReduce communication volume (proportional to model size) starts to dominate, especially across multi-node InfiniBand links. Small models on many GPUs hit this ceiling faster because the compute-to-communication ratio is lower.

**Gotcha: uneven batch sizes.** If your dataset doesn't divide evenly by the number of GPUs, the last GPU gets a smaller batch. DDP still AllReduces the gradients, so the last GPU's gradients are averaged with full-batch GPUs, introducing a slight bias. Use `drop_last=True` or pad to avoid this.

---

## 2. Fully Sharded Data Parallel (FSDP)

When DDP runs out of memory, FSDP is the next step. Based on Microsoft's ZeRO (Zero Redundancy Optimizer) paper, FSDP shards the model parameters, gradients, *and* optimizer states across all GPUs. Each GPU holds only 1/N of each tensor.

The trick: before computing a layer's forward pass, FSDP runs an AllGather to temporarily reconstruct the full layer's parameters on every GPU. After the backward pass, a ReduceScatter distributes each GPU's gradient shard back to its owner. The full-layer buffer is freed immediately — so only one layer's worth of temporary memory is used at any time.

<figure class="fig" id="fig-fsdp">
<h3>Figure 2 — Fully Sharded Data Parallel (FSDP / ZeRO-3)</h3>
<svg viewBox="0 0 460 260" width="100%">
  <text x="230" y="14" text-anchor="middle" fill="#6B7A94" font-size="9" font-weight="600">① Idle — each GPU holds 1/N of everything</text>
  <rect x="18" y="22" width="85" height="34" rx="5" fill="#CFDDFA" stroke="#4E6AD4" stroke-width="1"/>
  <text x="60" y="36" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 0</text>
  <text x="60" y="49" text-anchor="middle" fill="#6B7A94" font-size="7">Shard 0</text>
  <rect x="128" y="22" width="85" height="34" rx="5" fill="#DACBFA" stroke="#7556D0" stroke-width="1"/>
  <text x="170" y="36" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 1</text>
  <text x="170" y="49" text-anchor="middle" fill="#6B7A94" font-size="7">Shard 1</text>
  <rect x="238" y="22" width="85" height="34" rx="5" fill="#FADDC8" stroke="#C67E28" stroke-width="1"/>
  <text x="280" y="36" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 2</text>
  <text x="280" y="49" text-anchor="middle" fill="#6B7A94" font-size="7">Shard 2</text>
  <rect x="348" y="22" width="85" height="34" rx="5" fill="#C5ECD6" stroke="#24956E" stroke-width="1"/>
  <text x="390" y="36" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 3</text>
  <text x="390" y="49" text-anchor="middle" fill="#6B7A94" font-size="7">Shard 3</text>
  <text x="230" y="72" text-anchor="middle" fill="#7556D0" font-size="14">↓</text>
  <text x="230" y="86" text-anchor="middle" fill="#7556D0" font-size="9" font-weight="600">② All-Gather → reconstruct full layer on each GPU</text>
  <rect x="18" y="94" width="85" height="34" rx="5" fill="#EDEBFA" stroke="#7556D0" stroke-width="1"/>
  <text x="60" y="108" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 0</text>
  <text x="60" y="121" text-anchor="middle" fill="#7556D0" font-size="7">Full Layer</text>
  <rect x="128" y="94" width="85" height="34" rx="5" fill="#EDEBFA" stroke="#7556D0" stroke-width="1"/>
  <text x="170" y="108" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 1</text>
  <text x="170" y="121" text-anchor="middle" fill="#7556D0" font-size="7">Full Layer</text>
  <rect x="238" y="94" width="85" height="34" rx="5" fill="#EDEBFA" stroke="#7556D0" stroke-width="1"/>
  <text x="280" y="108" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 2</text>
  <text x="280" y="121" text-anchor="middle" fill="#7556D0" font-size="7">Full Layer</text>
  <rect x="348" y="94" width="85" height="34" rx="5" fill="#EDEBFA" stroke="#7556D0" stroke-width="1"/>
  <text x="390" y="108" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 3</text>
  <text x="390" y="121" text-anchor="middle" fill="#7556D0" font-size="7">Full Layer</text>
  <text x="230" y="145" text-anchor="middle" fill="#3B4562" font-size="14">↓</text>
  <text x="230" y="159" text-anchor="middle" fill="#3B4562" font-size="9" font-weight="600">③ Forward / Backward pass</text>
  <text x="230" y="175" text-anchor="middle" fill="#3B4562" font-size="14">↓</text>
  <text x="230" y="189" text-anchor="middle" fill="#C67E28" font-size="9" font-weight="600">④ Reduce-Scatter → each GPU keeps its own gradient shard</text>
  <rect x="18" y="197" width="85" height="34" rx="5" fill="#CFDDFA" stroke="#4E6AD4" stroke-width="1"/>
  <text x="60" y="211" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 0</text>
  <text x="60" y="224" text-anchor="middle" fill="#6B7A94" font-size="7">∇ Shard 0</text>
  <rect x="128" y="197" width="85" height="34" rx="5" fill="#DACBFA" stroke="#7556D0" stroke-width="1"/>
  <text x="170" y="211" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 1</text>
  <text x="170" y="224" text-anchor="middle" fill="#6B7A94" font-size="7">∇ Shard 1</text>
  <rect x="238" y="197" width="85" height="34" rx="5" fill="#FADDC8" stroke="#C67E28" stroke-width="1"/>
  <text x="280" y="211" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 2</text>
  <text x="280" y="224" text-anchor="middle" fill="#6B7A94" font-size="7">∇ Shard 2</text>
  <rect x="348" y="197" width="85" height="34" rx="5" fill="#C5ECD6" stroke="#24956E" stroke-width="1"/>
  <text x="390" y="211" text-anchor="middle" fill="#1A2138" font-size="9" font-weight="600">GPU 3</text>
  <text x="390" y="224" text-anchor="middle" fill="#6B7A94" font-size="7">∇ Shard 3</text>
  <text x="230" y="250" text-anchor="middle" fill="#6B7A94" font-size="8">Temporary full-layer buffer freed after each layer</text>
</svg>
<figcaption>FSDP shards parameters, gradients, and optimizer states across GPUs. Before each layer's forward pass, an All-Gather reconstructs the full parameters temporarily. After backward, a Reduce-Scatter returns each GPU's gradient shard.</figcaption>
</figure>

### Practical nuances

**The three ZeRO stages.** FSDP implements ZeRO-3 (full sharding), but the concept has three levels. ZeRO-1 shards only optimizer states (saving ~4× on Adam overhead). ZeRO-2 also shards gradients. ZeRO-3 shards everything including parameters. Each level adds communication but reduces memory. In practice, most people jump straight to ZeRO-3/FSDP because the memory savings compound dramatically — a 70B model that needs 1.12 TB with DDP fits in 70 GB per GPU with FSDP across 16 devices.

**Communication is 1.5× DDP, not 3×.** People sometimes assume FSDP must be much slower because it does AllGather + ReduceScatter instead of just AllReduce. In practice, the total communication volume is about 1.5× DDP's AllReduce, and modern NCCL implementations overlap the next layer's AllGather with the current layer's computation. The real overhead is typically 5–15%, not 50%.

**The temporary buffer spike.** During the AllGather, each GPU briefly holds a full layer's parameters in memory. For a model with very large layers (e.g., an 18432-dim Gemini layer), this temporary buffer can be significant. If you're right at the memory edge, this spike can cause OOM even though the steady-state usage fits. Watch `torch.cuda.max_memory_allocated()`, not just `memory_allocated()`.

**FSDP vs DeepSpeed ZeRO.** Both implement the same algorithm. PyTorch FSDP is native and tightly integrated with `torch.compile`. DeepSpeed ZeRO has been around longer and supports CPU offloading (ZeRO-Offload) and NVMe offloading (ZeRO-Infinity) for extreme memory savings at the cost of throughput. For most GPU-only training, native FSDP is the cleaner choice.

---

## 3. Pipeline Parallelism (PP)

Pipeline parallelism splits the model *vertically* — by layers. GPU 0 runs layers 0–7, GPU 1 runs layers 8–15, and so on. Each GPU passes its output activations to the next stage, like an assembly line.

<figure class="fig" id="fig-pp">
<h3>Figure 3 — Pipeline Parallelism</h3>
<svg viewBox="0 0 490 250" width="100%">
  <text x="245" y="14" text-anchor="middle" fill="#6B7A94" font-size="8">Model split by layers into sequential stages</text>
  <rect x="16" y="22" width="92" height="42" rx="6" fill="#CFDDFA" stroke="#C67E28" stroke-width="1.2"/>
  <text x="62" y="40" text-anchor="middle" fill="#C67E28" font-size="10" font-weight="600">Stage 0</text>
  <text x="62" y="55" text-anchor="middle" fill="#6B7A94" font-size="8">L0–7</text>
  <rect x="132" y="22" width="92" height="42" rx="6" fill="#DACBFA" stroke="#C67E28" stroke-width="1.2"/>
  <text x="178" y="40" text-anchor="middle" fill="#C67E28" font-size="10" font-weight="600">Stage 1</text>
  <text x="178" y="55" text-anchor="middle" fill="#6B7A94" font-size="8">L8–15</text>
  <rect x="248" y="22" width="92" height="42" rx="6" fill="#FADDC8" stroke="#C67E28" stroke-width="1.2"/>
  <text x="294" y="40" text-anchor="middle" fill="#C67E28" font-size="10" font-weight="600">Stage 2</text>
  <text x="294" y="55" text-anchor="middle" fill="#6B7A94" font-size="8">L16–23</text>
  <rect x="364" y="22" width="92" height="42" rx="6" fill="#C5ECD6" stroke="#C67E28" stroke-width="1.2"/>
  <text x="410" y="40" text-anchor="middle" fill="#C67E28" font-size="10" font-weight="600">Stage 3</text>
  <text x="410" y="55" text-anchor="middle" fill="#6B7A94" font-size="8">L24–31</text>
  <line x1="111" y1="43" x2="129" y2="43" stroke="#C67E28" stroke-width="1.3"/>
  <polygon points="129,43 123,40 123,46" fill="#C67E28"/>
  <text x="120" y="37" text-anchor="middle" fill="#C67E28" font-size="7">act</text>
  <line x1="227" y1="43" x2="245" y2="43" stroke="#C67E28" stroke-width="1.3"/>
  <polygon points="245,43 239,40 239,46" fill="#C67E28"/>
  <text x="236" y="37" text-anchor="middle" fill="#C67E28" font-size="7">act</text>
  <line x1="343" y1="43" x2="361" y2="43" stroke="#C67E28" stroke-width="1.3"/>
  <polygon points="361,43 355,40 355,46" fill="#C67E28"/>
  <text x="352" y="37" text-anchor="middle" fill="#C67E28" font-size="7">act</text>
  <text x="14" y="88" fill="#3B4562" font-size="9" font-weight="600">Schedule (time →)</text>
  <text x="8" y="115" fill="#6B7A94" font-size="9">S0</text>
  <rect x="28" y="98" width="46" height="26" rx="4" fill="#6E9CF0" opacity="0.75"/><text x="51" y="114" text-anchor="middle" fill="#fff" font-size="9" font-weight="600">Fwd</text>
  <rect x="228" y="98" width="46" height="26" rx="4" fill="#6E9CF0" opacity="0.3" stroke="#6E9CF0" stroke-width="1"/><text x="251" y="114" text-anchor="middle" fill="#1A2138" font-size="8">Bwd</text>
  <text x="8" y="147" fill="#6B7A94" font-size="9">S1</text>
  <rect x="28" y="130" width="46" height="26" rx="4" fill="#C44040" opacity="0.08" stroke="#C44040" stroke-width="0.6" stroke-dasharray="3,2"/><text x="51" y="146" text-anchor="middle" fill="#C44040" font-size="8" opacity="0.5">idle</text>
  <rect x="78" y="130" width="46" height="26" rx="4" fill="#A484E0" opacity="0.75"/><text x="101" y="146" text-anchor="middle" fill="#fff" font-size="9" font-weight="600">Fwd</text>
  <rect x="278" y="130" width="46" height="26" rx="4" fill="#A484E0" opacity="0.3" stroke="#A484E0" stroke-width="1"/><text x="301" y="146" text-anchor="middle" fill="#1A2138" font-size="8">Bwd</text>
  <text x="8" y="179" fill="#6B7A94" font-size="9">S2</text>
  <rect x="28" y="162" width="46" height="26" rx="4" fill="#C44040" opacity="0.08" stroke="#C44040" stroke-width="0.6" stroke-dasharray="3,2"/><text x="51" y="178" text-anchor="middle" fill="#C44040" font-size="8" opacity="0.5">idle</text>
  <rect x="78" y="162" width="46" height="26" rx="4" fill="#C44040" opacity="0.08" stroke="#C44040" stroke-width="0.6" stroke-dasharray="3,2"/><text x="101" y="178" text-anchor="middle" fill="#C44040" font-size="8" opacity="0.5">idle</text>
  <rect x="128" y="162" width="46" height="26" rx="4" fill="#E8A052" opacity="0.75"/><text x="151" y="178" text-anchor="middle" fill="#fff" font-size="9" font-weight="600">Fwd</text>
  <rect x="328" y="162" width="46" height="26" rx="4" fill="#E8A052" opacity="0.3" stroke="#E8A052" stroke-width="1"/><text x="351" y="178" text-anchor="middle" fill="#1A2138" font-size="8">Bwd</text>
  <text x="8" y="211" fill="#6B7A94" font-size="9">S3</text>
  <rect x="28" y="194" width="46" height="26" rx="4" fill="#C44040" opacity="0.08" stroke="#C44040" stroke-width="0.6" stroke-dasharray="3,2"/><text x="51" y="210" text-anchor="middle" fill="#C44040" font-size="8" opacity="0.5">idle</text>
  <rect x="78" y="194" width="46" height="26" rx="4" fill="#C44040" opacity="0.08" stroke="#C44040" stroke-width="0.6" stroke-dasharray="3,2"/><text x="101" y="210" text-anchor="middle" fill="#C44040" font-size="8" opacity="0.5">idle</text>
  <rect x="128" y="194" width="46" height="26" rx="4" fill="#C44040" opacity="0.08" stroke="#C44040" stroke-width="0.6" stroke-dasharray="3,2"/><text x="151" y="210" text-anchor="middle" fill="#C44040" font-size="8" opacity="0.5">idle</text>
  <rect x="178" y="194" width="46" height="26" rx="4" fill="#5CC48E" opacity="0.75"/><text x="201" y="210" text-anchor="middle" fill="#fff" font-size="9" font-weight="600">Fwd</text>
  <rect x="378" y="194" width="46" height="26" rx="4" fill="#5CC48E" opacity="0.3" stroke="#5CC48E" stroke-width="1"/><text x="401" y="210" text-anchor="middle" fill="#1A2138" font-size="8">Bwd</text>
  <text x="28" y="238" fill="#C44040" font-size="8">■ idle = pipeline bubble</text>
  <text x="190" y="238" fill="#6B7A94" font-size="8">Micro-batching overlaps work to fill gaps</text>
</svg>
<figcaption>Pipeline parallelism splits the model by layers into stages. Activations flow stage-to-stage. The "bubble" slots show idle time — later stages must wait for earlier ones to finish. Micro-batching fills these gaps by overlapping multiple mini-batches.</figcaption>
</figure>

### Practical nuances

**The bubble tax is steep.** With 4 pipeline stages and 4 micro-batches, the bubble fraction is (4-1)/(4+4-1) = 43% — nearly half your compute is wasted. With 8 micro-batches it's 30%, and with 16 it's 19%. You need at least 4× as many micro-batches as stages to get the bubble below 20%. This is why pipeline parallelism is rarely used alone — it's almost always combined with data parallelism (to increase the micro-batch count) or tensor parallelism (to reduce the stage count).

**Memory isn't balanced.** The first and last pipeline stages often use more memory than middle stages. The first stage stores the embedding layer's activations for all micro-batches in flight. The last stage stores the loss computation and the start of the backward pass. If you're OOM on stage 0 but have headroom on stage 2, you're pipeline-imbalanced and losing efficiency.

**Point-to-point communication is cheap.** Unlike DDP's AllReduce (which moves the entire model's gradients), PP only sends activation tensors between adjacent stages. For a 4096-dimensional transformer, that's just `batch_size × seq_len × 4096 × 2 bytes` per micro-batch — orders of magnitude less than a full gradient AllReduce. This makes PP work well even over relatively slow inter-node InfiniBand links.

**1F1B schedule.** The naïve GPipe schedule shown above runs all forward passes before all backward passes, maximizing the activation memory (all micro-batches' activations stored simultaneously). The PipeDream-Flush/1F1B schedule interleaves forward and backward passes: after the pipeline fills, each stage alternates one-forward-one-backward, freeing activations earlier and reducing peak memory by ~(stages-1)/stages.

---

## 4. Tensor Parallelism (TP)

Tensor parallelism goes *inside* individual layers. Instead of assigning whole layers to different GPUs, it splits each weight matrix across GPUs. For a linear layer `Y = X·W`, the weight `W` is split column-wise so GPU `i` holds `W[:,i]` and computes `X·W[:,i]`. An AllReduce then sums the partial results across GPUs to produce the full output.

<figure class="fig" id="fig-tp">
<h3>Figure 4 — Tensor Parallelism</h3>
<svg viewBox="0 0 440 230" width="100%">
  <rect x="176" y="4" width="88" height="28" rx="5" fill="#ECF0F6" stroke="#DDE3ED" stroke-width="1"/>
  <text x="220" y="22" text-anchor="middle" fill="#1A2138" font-size="11" font-weight="600">Input X</text>
  <line x1="220" y1="34" x2="61" y2="56" stroke="#9AA5B8" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="220" y1="34" x2="167" y2="56" stroke="#9AA5B8" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="220" y1="34" x2="273" y2="56" stroke="#9AA5B8" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="220" y1="34" x2="379" y2="56" stroke="#9AA5B8" stroke-width="1" stroke-dasharray="3,2"/>
  <rect x="18" y="58" width="86" height="50" rx="6" fill="#CFDDFA" stroke="#4E6AD4" stroke-width="1.2"/>
  <text x="61" y="74" text-anchor="middle" fill="#1A2138" font-size="10" font-weight="600">GPU 0</text>
  <text x="61" y="89" text-anchor="middle" fill="#4E6AD4" font-size="8">X · W[:,0]</text>
  <text x="61" y="102" text-anchor="middle" fill="#6B7A94" font-size="7">partial Y0</text>
  <rect x="124" y="58" width="86" height="50" rx="6" fill="#DACBFA" stroke="#7556D0" stroke-width="1.2"/>
  <text x="167" y="74" text-anchor="middle" fill="#1A2138" font-size="10" font-weight="600">GPU 1</text>
  <text x="167" y="89" text-anchor="middle" fill="#7556D0" font-size="8">X · W[:,1]</text>
  <text x="167" y="102" text-anchor="middle" fill="#6B7A94" font-size="7">partial Y1</text>
  <rect x="230" y="58" width="86" height="50" rx="6" fill="#FADDC8" stroke="#C67E28" stroke-width="1.2"/>
  <text x="273" y="74" text-anchor="middle" fill="#1A2138" font-size="10" font-weight="600">GPU 2</text>
  <text x="273" y="89" text-anchor="middle" fill="#C67E28" font-size="8">X · W[:,2]</text>
  <text x="273" y="102" text-anchor="middle" fill="#6B7A94" font-size="7">partial Y2</text>
  <rect x="336" y="58" width="86" height="50" rx="6" fill="#C5ECD6" stroke="#24956E" stroke-width="1.2"/>
  <text x="379" y="74" text-anchor="middle" fill="#1A2138" font-size="10" font-weight="600">GPU 3</text>
  <text x="379" y="89" text-anchor="middle" fill="#24956E" font-size="8">X · W[:,3]</text>
  <text x="379" y="102" text-anchor="middle" fill="#6B7A94" font-size="7">partial Y3</text>
  <line x1="61" y1="110" x2="220" y2="140" stroke="#24956E" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="167" y1="110" x2="220" y2="140" stroke="#24956E" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="273" y1="110" x2="220" y2="140" stroke="#24956E" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="379" y1="110" x2="220" y2="140" stroke="#24956E" stroke-width="1" stroke-dasharray="3,2"/>
  <rect x="166" y="142" width="108" height="26" rx="5" fill="#D8F0E4" stroke="#24956E" stroke-width="1.2"/>
  <text x="220" y="159" text-anchor="middle" fill="#24956E" font-size="10" font-weight="700">AllReduce → Y</text>
  <rect x="176" y="180" width="88" height="26" rx="5" fill="#D8F0E4" stroke="#24956E" stroke-width="1"/>
  <text x="220" y="197" text-anchor="middle" fill="#1A2138" font-size="11" font-weight="600">Y = X · W</text>
  <text x="220" y="222" text-anchor="middle" fill="#6B7A94" font-size="8">Repeated every layer — requires NVLink speed</text>
</svg>
<figcaption>Tensor parallelism splits weight matrices column-wise across GPUs. Each GPU computes a partial product X·W[:,i], then an AllReduce sums the results to produce the full output Y. This happens for every layer, demanding NVLink-class interconnect bandwidth.</figcaption>
</figure>

### Practical nuances

**NVLink or nothing.** Every layer in the transformer triggers two AllReduce operations (one in the attention block, one in the MLP). For a 96-layer GPT-3, that's 192 AllReduces per iteration. Over NVLink (600–900 GB/s), this is fast. Over InfiniBand (50–100 GB/s), it's a disaster. This is why TP is almost exclusively used *within* a single node (8 GPUs on NVLink) and almost never across nodes.

**TP > 8 rarely makes sense.** Since DGX/HGX nodes have 8 GPUs connected by NVLink, and cross-node links are 6–10× slower, TP degree is almost always capped at 8 (or the number of GPUs per node). Going beyond 8 means crossing the node boundary, which tanks throughput.

**Attention heads map cleanly to TP.** In multi-head attention, each head is an independent linear transform. With 32 heads and TP=8, each GPU computes 4 heads — a clean split with no cross-GPU dependencies until the final output projection. This is why TP maps naturally to transformers.

**Sequence parallelism (SP).** Standard TP splits weight matrices but replicates activation tensors — the input `X` is the same on every GPU. Sequence parallelism (introduced in Megatron-LM v3) extends TP to also split activations along the sequence dimension in the non-tensor-parallel regions (LayerNorm, dropout). This reduces activation memory by TP degree and is essentially free to implement if you already have TP. Always turn it on.

---

## 5. 3D Parallelism: Putting It Together

For models that are both wide and deep and need hundreds of GPUs, no single strategy works alone. 3D parallelism combines all three: TP within a node, PP across node groups, and DP (or FSDP) across the remaining dimension.

<figure class="fig" id="fig-3d">
<h3>Figure 5 — 3D Parallelism (TP + PP + DP)</h3>
<svg viewBox="0 0 450 195" width="100%">
  <rect x="8" y="28" width="130" height="100" rx="8" fill="#E2F2E8" stroke="#24956E" stroke-width="1" stroke-dasharray="5,3"/>
  <text x="73" y="20" text-anchor="middle" fill="#24956E" font-size="9" font-weight="600">TP (intra-node)</text>
  <rect x="20" y="44" width="24" height="24" rx="4" fill="#B8EAD0" stroke="#24956E" stroke-width="0.8"/><text x="32" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">0</text>
  <rect x="50" y="44" width="24" height="24" rx="4" fill="#B8EAD0" stroke="#24956E" stroke-width="0.8"/><text x="62" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">1</text>
  <rect x="80" y="44" width="24" height="24" rx="4" fill="#B8EAD0" stroke="#24956E" stroke-width="0.8"/><text x="92" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">2</text>
  <rect x="110" y="44" width="24" height="24" rx="4" fill="#B8EAD0" stroke="#24956E" stroke-width="0.8"/><text x="122" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">3</text>
  <text x="73" y="90" text-anchor="middle" fill="#6B7A94" font-size="8">Split matrices</text>
  <text x="73" y="104" text-anchor="middle" fill="#6B7A94" font-size="8">AllReduce/layer</text>
  <rect x="158" y="28" width="130" height="100" rx="8" fill="#FEF3E2" stroke="#C67E28" stroke-width="1" stroke-dasharray="5,3"/>
  <text x="223" y="20" text-anchor="middle" fill="#C67E28" font-size="9" font-weight="600">PP (across nodes)</text>
  <rect x="170" y="44" width="24" height="24" rx="4" fill="#FADDC8" stroke="#C67E28" stroke-width="0.8"/><text x="182" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">S0</text>
  <rect x="200" y="44" width="24" height="24" rx="4" fill="#FADDC8" stroke="#C67E28" stroke-width="0.8"/><text x="212" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">S1</text>
  <rect x="230" y="44" width="24" height="24" rx="4" fill="#FADDC8" stroke="#C67E28" stroke-width="0.8"/><text x="242" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">S2</text>
  <rect x="260" y="44" width="24" height="24" rx="4" fill="#FADDC8" stroke="#C67E28" stroke-width="0.8"/><text x="272" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">S3</text>
  <text x="223" y="90" text-anchor="middle" fill="#6B7A94" font-size="8">Split by layers</text>
  <text x="223" y="104" text-anchor="middle" fill="#6B7A94" font-size="8">P2P activations</text>
  <rect x="308" y="28" width="130" height="100" rx="8" fill="#E8EBF6" stroke="#4E6AD4" stroke-width="1" stroke-dasharray="5,3"/>
  <text x="373" y="20" text-anchor="middle" fill="#4E6AD4" font-size="9" font-weight="600">DP / FSDP</text>
  <rect x="320" y="44" width="44" height="24" rx="4" fill="#CFDDFA" stroke="#4E6AD4" stroke-width="0.8"/><text x="342" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">R0</text>
  <rect x="372" y="44" width="44" height="24" rx="4" fill="#CFDDFA" stroke="#4E6AD4" stroke-width="0.8"/><text x="394" y="60" text-anchor="middle" fill="#1A2138" font-size="8" font-weight="600">R1</text>
  <text x="373" y="90" text-anchor="middle" fill="#6B7A94" font-size="8">Duplicate pipeline</text>
  <text x="373" y="104" text-anchor="middle" fill="#6B7A94" font-size="8">AllReduce grads</text>
  <line x1="140" y1="78" x2="156" y2="78" stroke="#9AA5B8" stroke-width="1.2"/><polygon points="156,78 150,75 150,81" fill="#9AA5B8"/>
  <line x1="290" y1="78" x2="306" y2="78" stroke="#9AA5B8" stroke-width="1.2"/><polygon points="306,78 300,75 300,81" fill="#9AA5B8"/>
  <rect x="143" y="144" width="164" height="24" rx="5" fill="#F4F6FB" stroke="#DDE3ED" stroke-width="1"/>
  <text x="225" y="160" text-anchor="middle" fill="#1A2138" font-size="10" font-weight="600">TP × PP × DP = Total GPUs</text>
  <text x="225" y="186" text-anchor="middle" fill="#6B7A94" font-size="8">e.g. 8 × 4 × 2 = 64 GPUs</text>
</svg>
<figcaption>3D parallelism combines all three strategies: Tensor Parallel within a node (NVLink), Pipeline Parallel across node groups (InfiniBand), and Data Parallel for throughput scaling. The product of all three degrees equals the total GPU count.</figcaption>
</figure>

### Practical nuances

**The decomposition recipe.** The standard approach for clusters with 8-GPU nodes: TP = 8 (fill the node with tensor parallelism over NVLink), PP = total_nodes / DP (split the model depth across groups of nodes), DP = whatever's left (for data-parallel throughput and gradient averaging). For a 128-GPU cluster (16 nodes) training a 70B model: TP=8, PP=4, DP=4 gives 8×4×4 = 128 GPUs.

**Each dimension has a different cost.** TP communication is frequent (every layer) but fast (NVLink). PP communication is rare (once per micro-batch per stage) but serializes the pipeline. DP communication is once per step but moves the full gradient volume. The goal is to minimize the bottleneck dimension. If your interconnect is slow, maximize PP (least bandwidth needed) and minimize TP.

**Debugging is hard.** A bug in 3D parallelism training could be caused by incorrect sharding (TP), activation mismatch between stages (PP), or gradient desync (DP). Each dimension introduces its own category of correctness issues. Start with TP only (single-node), add PP (multi-node), then add DP — don't try to debug all three simultaneously.

---

## When to Use What

<style>
.decision-table{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0;}
.decision-table th{background:#F4F6FB;color:#3B4562;font-weight:600;text-align:left;padding:9px 12px;border:1px solid #DDE3ED;}
.decision-table td{padding:8px 12px;border:1px solid #DDE3ED;vertical-align:top;}
.decision-table tr:nth-child(even){background:#FAFBFD;}
.decision-table tr:hover{background:#F0F4FF;}
</style>

<table class="decision-table">
<thead>
<tr><th>Your situation</th><th>Start with</th><th>Why</th></tr>
</thead>
<tbody>
<tr><td>Model fits on one GPU</td><td>DDP</td><td>Maximum simplicity, near-linear scaling</td></tr>
<tr><td>Model is 2–5× GPU memory</td><td>FSDP (ZeRO-3)</td><td>Shards everything, minimal code changes from DDP</td></tr>
<tr><td>Model is 10–50× GPU memory</td><td>FSDP + TP</td><td>TP reduces per-layer memory, FSDP handles the rest</td></tr>
<tr><td>Very deep model, slow interconnect</td><td>PP + DP</td><td>PP needs minimal bandwidth</td></tr>
<tr><td>Frontier model (100B+), fast cluster</td><td>3D (TP+PP+FSDP)</td><td>The only way to fit, the only way to be efficient</td></tr>
<tr><td>Inference at scale</td><td>TP (or TP+PP)</td><td>No gradients/optimizer — just split the weights</td></tr>
</tbody>
</table>

The heuristic: start with the simplest strategy that fits your model in memory. Only add complexity (more parallelism dimensions) when forced to by memory or throughput constraints. Every additional dimension introduces communication overhead, debugging surface area, and configuration tuning.

---

## Memory Cheat Sheet

Training memory per parameter with mixed-precision Adam:

<table class="decision-table">
<thead>
<tr><th>Component</th><th>Bytes/param</th><th>Sharded by</th></tr>
</thead>
<tbody>
<tr><td>FP16 parameters</td><td>2</td><td>TP, PP</td></tr>
<tr><td>FP16 gradients</td><td>2</td><td>TP, PP, DP (FSDP)</td></tr>
<tr><td>FP32 master weights</td><td>4</td><td>DP (FSDP)</td></tr>
<tr><td>FP32 Adam momentum</td><td>4</td><td>DP (FSDP)</td></tr>
<tr><td>FP32 Adam variance</td><td>4</td><td>DP (FSDP)</td></tr>
<tr><td><strong>Total</strong></td><td><strong>16</strong></td><td>—</td></tr>
</tbody>
</table>

So a 70B model needs 70 × 16 = **1.12 TB** of GPU memory for training (before activations). On 8× 80 GB GPUs (640 GB total), DDP can't even fit the model. FSDP brings it to 1.12 TB / 8 = 140 GB total, which is 17.5 GB per GPU for model state, leaving plenty of room for activations.

---

## Interactive Parallelism Explorer

Pick a model, pick a GPU cluster, switch between paradigms, and watch the memory bars and efficiency metrics update in real time. Run the training simulation to see GPU temperatures and memory fluctuate through forward/backward/communication phases.

<div id="parallelism-root"><div class="demo-loading">Loading Parallelism Explorer…</div></div>

---

*Further reading: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965).*
