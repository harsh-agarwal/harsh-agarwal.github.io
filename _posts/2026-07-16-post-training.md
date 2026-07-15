---
layout: playground-post
title: "How Language Models Learn to Behave: An Interactive Guide to RLHF"
date: 2026-07-16
description: "An interactive deep-dive into RLHF, reward modeling, DPO, Constitutional AI, and the full post-training stack — with live playgrounds for each concept."
tags: [machine-learning, rlhf, alignment, llm, post-training]
---

A model trained purely on next-token prediction learns to **imitate text**, not to **behave helpfully**. These are very different objectives — and closing that gap is exactly what post-training does.

This post walks through each stage of the modern post-training stack: Supervised Fine-Tuning, reward modeling, PPO-based RLHF, Direct Preference Optimization, and Constitutional AI. Every concept has a live playground beneath it so you can build real intuition, not just follow equations.

<style>
.fig-note {
  font-size: 13px;
  color: #777;
  font-style: italic;
  margin: -4px 0 20px;
  line-height: 1.6;
}
.fig-iframe {
  border-radius: 10px;
  display: block;
  margin: 24px 0;
  width: 100%;
  border: none;
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

  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || typeof d.iframeId !== 'string' || typeof d.height !== 'number') return;
    var frame = document.getElementById(d.iframeId);
    if (!frame) return;
    frame.style.height = (d.height + 24) + 'px';
  });
});
</script>

---

## 01 — Why Pre-Training Isn't Enough

Pre-training gives a model encyclopedic knowledge and fluency. It also gives it an uncensored mirror of the internet: toxic content, misinformation, contradictions, and no concept of what a user actually *wants*. Post-training steers the model toward outputs that humans genuinely value.

The playground below shows how the model's **next-token distribution** changes at each stage for a fixed prompt. Toggle between stages to see which continuations become more or less probable.

<iframe id="rlhf-w1"
  src="/playgrounds/post-training/w1-distribution-shift.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## 02 — Supervised Fine-Tuning (SFT)

SFT trains the model on curated **(prompt → ideal response)** pairs written by human experts. It works well but has a ceiling: humans can write good responses, but writing the *optimal* one from scratch is genuinely hard — and doing it at scale is expensive.

> **Key insight:** It is much easier for a human to **compare** two answers and say "this one is better" than to **write** the perfect answer from scratch. This observation motivates everything that follows.

---

## 03 — Training a Reward Model

The reward model (RM) turns subjective human judgments into a **differentiable scalar signal** the optimizer can train against. It learns from pairwise preferences using a ranking loss:

```
P(y_w preferred over y_l) = σ( r(x, y_w) − r(x, y_l) )

Loss = −E[ log σ( r(x, y_w) − r(x, y_l) ) ]

where:
  r(x, y) = reward score for completion y given prompt x
  y_w     = preferred (winner) response
  y_l     = dispreferred (loser) response
  σ       = sigmoid function
```

The playground below puts you in the annotator's seat. Click the response you prefer and watch the reward model's training loss curve update in real time.

<iframe id="rlhf-w2"
  src="/playgrounds/post-training/w2-preference-labeling.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## 04 — PPO and the KL Penalty

Once we have a reward model we fine-tune the language model with **Proximal Policy Optimization (PPO)**. The model generates completions, the RM scores them, and gradients update the policy.

But an unconstrained optimizer will find *exploits* — outputs that score highly without being genuinely good. The fix is a **KL divergence penalty** that keeps the fine-tuned policy tethered to the original SFT model:

```
Objective = E[ r(x,y) ] − β · KL( π_RL(y|x) ‖ π_SFT(y|x) )

  r(x,y)   = reward model score
  β         = KL penalty coefficient (typically 0.02 – 0.2)
  π_RL      = current (fine-tuning) policy
  π_SFT     = frozen reference (SFT) policy
  KL(·‖·)  = KL divergence — how far the policies have drifted
```

The β coefficient controls the fundamental tradeoff: **exploit the reward signal vs. stay safe near the reference.**

<iframe id="rlhf-w3"
  src="/playgrounds/post-training/w3-kl-tug-of-war.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## 05 — Reward Hacking

The reward model is an imperfect proxy for human preferences. A powerful optimizer will eventually find ways to score highly that don't correspond to genuinely good outputs. This is **reward hacking**, one of the central failure modes of RLHF.

> **Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

Drag the slider below and watch the gap open between what the reward model reports and what is actually happening to output quality.

<iframe id="rlhf-w4"
  src="/playgrounds/post-training/w4-reward-hacking.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## 06 — Direct Preference Optimization (DPO)

**DPO** is an elegant insight: the RL objective with a KL constraint has a closed-form optimal policy, which means you can derive a training objective directly from preference data — no reward model training, no RL loop, no PPO at all.

```
L_DPO = −E[(x, y_w, y_l)] · log σ(
    β · log( π_θ(y_w|x) / π_ref(y_w|x) )
  − β · log( π_θ(y_l|x) / π_ref(y_l|x) )
)

Intuition:
  → Increase probability of preferred responses relative to the reference model
  → Decrease probability of dispreferred responses relative to the reference model
  → β controls how far we can deviate from the reference
```

The pipeline comparison below shows where the complexity lives in each approach.

<iframe id="rlhf-w5"
  src="/playgrounds/post-training/w5-rlhf-vs-dpo.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## 07 — Constitutional AI & RLAIF

Generating human preference labels is expensive. **Constitutional AI (CAI)** replaces much of this with AI-generated feedback guided by a written set of principles — the "constitution."

The model critiques its own outputs, revises them, and the resulting (original → revised) pairs become preference training data. At scale, an AI evaluator generates preference labels directly — this is **RLAIF (RL from AI Feedback)**.

> **The loop:** Generate → Critique against the constitution → Revise → Use (original, revised) as a preference pair in the training set.

<iframe id="rlhf-w6"
  src="/playgrounds/post-training/w6-constitutional-ai.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## 08 — Why One Round Is Never Enough

Modern post-training runs **multiple iterations**. Each round generates completions from the *current* model, collects preferences on those specific completions, and updates the model again. This matters because the model's distribution shifts — preferences collected on the SFT model become stale once the model has evolved.

Each round can target a different dimension: helpfulness, safety, honesty, factuality. Step through the rounds below to see how targeted training affects each axis, and notice the diminishing returns curve as training matures.

<iframe id="rlhf-w7"
  src="/playgrounds/post-training/w7-iterative-training.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## 09 — The Full Post-Training Stack

The diagram below shows how all these components connect. The main flow runs left to right. DPO provides a shortcut skipping the reward model and RL loop. Iterative rounds feed the aligned model back into fresh preference collection. Constitutional AI / RLAIF replaces expensive human labels with AI-generated feedback at scale.

<iframe id="rlhf-w8"
  src="/playgrounds/post-training/w8-full-pipeline.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## The Core Tension

Every design decision in post-training is, at some level, a response to the same problem:

> **How do we make the training signal accurate enough that the optimizer improves the model rather than gaming the metric?**

SFT is limited by the quality of demonstrations. Reward models are limited by the quality of preference labels. DPO is limited by the diversity of the offline dataset. Constitutional AI is limited by the quality of the constitution. Iterative training helps, but diminishing returns set in. Each technique is a different attempt to approximate what humans value faithfully enough that gradient descent moves in the right direction.

---

## Key Papers

| Year | Paper | Contribution |
|------|-------|--------------|
| 2017 | Christiano et al. — *Deep RL from Human Preferences* | Original RLHF framework |
| 2020 | Stiennon et al. — *Learning to summarize with human feedback* | RLHF applied to text generation |
| 2022 | Ouyang et al. — *InstructGPT* | RLHF applied to large language models |
| 2022 | Bai et al. (Anthropic) — *Constitutional AI* | CAI and RLAIF |
| 2017 | Schulman et al. — *Proximal Policy Optimization* | The PPO algorithm |
| 2023 | Rafailov et al. — *Direct Preference Optimization* | DPO algorithm and theory |
| 2023 | Lightman et al. (OpenAI) — *Let's Verify Step by Step* | Process reward models |
