---
layout: playground-post
title: "How Language Models Learn to Behave: An Interactive Guide to RLHF"
date: 2026-07-16
description: "An interactive deep-dive into RLHF, reward modeling, DPO, Constitutional AI, and the full post-training stack, with live playgrounds for each concept."
tags: [machine-learning, rlhf, alignment, llm, post-training]
---

A model trained purely on next-token prediction learns to **imitate text**, not to **behave helpfully**. These are very different objectives, and closing that gap is exactly what post-training does.

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
.papers-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  margin: 16px 0 8px;
}
.papers-table th {
  text-align: left;
  padding: 8px 14px 8px 0;
  border-bottom: 2px solid #e0dcd4;
  font-weight: 600;
  color: #444;
  white-space: nowrap;
}
.papers-table td {
  padding: 9px 14px 9px 0;
  border-bottom: 1px solid #f0ede8;
  vertical-align: top;
  line-height: 1.5;
}
.papers-table td:first-child {
  white-space: nowrap;
  color: #999;
  font-size: 13px;
  padding-right: 20px;
}
.papers-table td:last-child {
  color: #666;
}
</style>

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['\\[', '\\]']]
  }
};
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

## Why Pre-Training Isn't Enough

If you train a language model purely on next-token prediction over internet text, you end up with something that is simultaneously impressive and deeply unreliable. It can write, reason, summarize, and converse. It can also confidently produce misinformation, generate harmful content without hesitation, and completely misread what you were actually asking for.

The problem isn't capability. It's that "predict the next token on internet text" and "be genuinely helpful to a specific person" are not the same objective. Pre-training makes the model fluent. Post-training makes it useful.

The playground below makes this concrete. Pick a prompt and watch how the probability mass over next tokens shifts at each training stage. The pre-trained model's distribution looks very different from the final aligned model's, and understanding that shift is the whole story.

<iframe id="rlhf-w1"
  src="/playgrounds/post-training/w1-distribution-shift.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## Supervised Fine-Tuning: Teaching by Example

The first step toward useful behavior is Supervised Fine-Tuning. You take the pre-trained model and continue training it on a curated dataset of (prompt, ideal response) pairs written by human experts. The model learns the format, style, and structure of genuinely helpful responses rather than just predicting what would come next in generic web text.

SFT works well, but it has a real ceiling. Writing the *optimal* response to a difficult question is genuinely hard, even for experts. It requires knowing not just what to say, but how to say it for this particular model and context. Annotation at that quality doesn't scale, and the model is fundamentally bounded by the quality of the demonstrations it learns from.

Here's the insight that unlocks everything that follows: **it is much easier for a human to compare two responses and say "this one is better" than to write the optimal one from scratch.** Judgment is cheaper than generation. The rest of the post-training stack is built on this observation.

---

## Training a Reward Model

Since humans can't be in the loop for every gradient update (a single training run might process millions of examples), we need a way to convert human judgments into a signal the optimizer can use automatically. That's the job of the reward model.

The reward model learns from pairwise comparisons. You show it a prompt alongside two responses, and it learns to predict which one a human annotator would prefer. The training objective encodes exactly this:

<div>\[P(y_w \succ y_l) = \sigma\!\left( r(x,\, y_w) - r(x,\, y_l) \right)\]</div>

<div>\[\mathcal{L}_{\text{RM}} = -\,\mathbb{E}\!\left[ \log \sigma\!\left( r(x,\, y_w) - r(x,\, y_l) \right) \right]\]</div>

<div>\[\begin{array}{r@{\;{:}\;}l}
r(x, y) & \text{reward score for response } y \text{ given prompt } x \\[5pt]
y_w & \text{preferred (winner) response} \\[5pt]
y_l & \text{dispreferred (loser) response} \\[5pt]
\sigma & \text{sigmoid function}
\end{array}\]</div>

Once trained, the reward model is a standalone function you can query cheaply at inference time. Give it any (prompt, response) pair and it returns a scalar score. You've distilled human preference into something differentiable, and that's what makes optimization against it possible.

The playground below puts you in the annotator's seat. Label a few pairs yourself and watch the reward model's training loss update as it learns from your choices.

<iframe id="rlhf-w2"
  src="/playgrounds/post-training/w2-preference-labeling.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## PPO and the KL Penalty

With a reward model in hand, you can now fine-tune the language model against it directly. The model generates completions, the reward model scores them, and gradients flow backward through the reward signal. This is **Proximal Policy Optimization (PPO)**, borrowed from robotics RL and adapted for language.

The setup works, with one important catch. A sufficiently powerful optimizer will find ways to score highly on the reward model that have nothing to do with being genuinely helpful. It might learn that verbose, confident-sounding responses get high scores, or discover quirks in how the reward model was trained and exploit those patterns relentlessly. Left unconstrained, the model would eventually produce responses that look great to the reward model and are useless to a real user.

The fix is a **KL divergence penalty**: a term in the objective that penalizes the policy for drifting too far from the original SFT model. Think of it as a leash. A long leash (low β) gives the model freedom to exploit the reward signal aggressively. A short leash (high β) keeps it close to safe, human-written behavior. The full objective is:

<div>\[\mathcal{J}_{\text{PPO}} = \mathbb{E}\!\left[ r(x,y) \right] - \beta \cdot \mathrm{KL}\!\left( \pi_{\text{RL}}(y|x) \;\Big\|\; \pi_{\text{SFT}}(y|x) \right)\]</div>

<div>\[\begin{array}{r@{\;{:}\;}l}
r(x,y) & \text{reward model score} \\[5pt]
\beta & \text{KL penalty coefficient (typically 0.02 to 0.2)} \\[5pt]
\pi_{\text{RL}} & \text{current (fine-tuning) policy} \\[5pt]
\pi_{\text{SFT}} & \text{frozen reference (SFT) policy} \\[5pt]
\mathrm{KL}(\cdot\|\cdot) & \text{KL divergence, measuring policy drift}
\end{array}\]</div>

Tune β below and watch the tug-of-war between reward exploitation and reference fidelity in real time.

<iframe id="rlhf-w3"
  src="/playgrounds/post-training/w3-kl-tug-of-war.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## Reward Hacking

Even with the KL penalty in place, the reward model is still just a proxy for human preferences. It was trained on a finite set of comparisons and learned to generalize. And any learned generalization can be exploited.

**Reward hacking** is what happens when the optimizer finds inputs that fool the reward model into high scores without achieving the underlying goal. Responses that are verbose in exactly the way the reward model rewards. Formatting patterns that score well but annoy real users. Confident assertions about facts the reward model can't verify.

> **Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

This isn't a fixable bug so much as a fundamental tension. The gap between "what the reward model scores highly" and "what humans actually want" is always present; the question is whether the gap is small enough to matter in practice. Drag the slider below and watch that gap widen as training continues beyond the sweet spot.

<iframe id="rlhf-w4"
  src="/playgrounds/post-training/w4-reward-hacking.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## DPO: Cutting Out the Middleman

RLHF with PPO works, but the pipeline is complex. Training a separate reward model, running PPO, managing the KL penalty, keeping multiple models in memory simultaneously — at large scale, this is expensive and finicky.

**Direct Preference Optimization (DPO)** found a cleaner path. It turns out that the RL-with-KL-constraint objective has a closed-form optimal policy, which means you can derive a training objective directly from preference pairs, with no reward model and no RL loop required. The math works out to:

<div>\[\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,\, y_w,\, y_l)}\!\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]\]</div>

<div>\[\begin{array}{r@{\;{:}\;}l}
\pi_\theta & \text{policy being trained} \\[5pt]
\pi_{\text{ref}} & \text{frozen reference (SFT) policy} \\[5pt]
\beta & \text{deviation budget, same role as in PPO}
\end{array}\]</div>

The β coefficient is doing the same job as in PPO, just expressed directly in terms of policy log-ratios rather than reward scores. The pipeline comparison below shows exactly how much complexity DPO eliminates compared to the full RLHF setup.

<iframe id="rlhf-w5"
  src="/playgrounds/post-training/w5-rlhf-vs-dpo.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## Constitutional AI and Learning from AI Feedback

Every technique described so far has quietly assumed a steady supply of human annotators. Someone writes the SFT demonstrations. Someone labels the preference pairs. At the scale modern LLMs require, this is genuinely expensive and slow, often the binding constraint on how much training you can do.

**Constitutional AI (CAI)**, developed at Anthropic, offers a partial escape. Instead of having humans label every preference pair, you give the model a written set of principles (be helpful, be honest, avoid harm) and have it critique its own outputs against those principles. The revised response becomes the preferred one; the original becomes the dispreferred one. You generate synthetic preference data automatically, at scale, without needing a human to evaluate each pair.

Push this further and you get **RLAIF (RL from AI Feedback)**, where a separate AI evaluator generates the preference labels entirely. The feedback loop runs without humans in the critical path at all. This can produce training data orders of magnitude faster than human annotation, at a fraction of the cost.

The tradeoff is fidelity. The quality of the training signal depends entirely on the quality of the constitution and the AI evaluator. If the constitution is vague or the evaluator is poorly calibrated, the training signal gets noisy. But when it's done well, CAI and RLAIF unlock the ability to run far more training rounds than human annotation alone could ever support.

<iframe id="rlhf-w6"
  src="/playgrounds/post-training/w6-constitutional-ai.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## Why One Round Is Never Enough

There's a subtle problem with running RLHF once and calling it done. The preference data you collected was labeled on responses from the SFT model. After one round of training, the model has changed: its distribution is different, the responses it generates look different, and the preferences collected on the old model are now stale. You're training a new model on data that reflects a model that no longer exists.

Modern post-training pipelines deal with this by running **multiple iterations**. Each round generates fresh completions from the current model, collects preference labels on those specific outputs, and updates. The training signal stays fresh because it's always reflecting the model as it actually is right now, not as it was before the last update.

Each round can also be targeted at a specific dimension: one round focused on helpfulness, the next on safety, the next on factual accuracy. The improvements diminish across rounds, but early rounds often produce the largest gains, and targeted iteration is how you shape the model across multiple axes at once. Step through the rounds below to see how each pass shifts the model's behavior.

<iframe id="rlhf-w7"
  src="/playgrounds/post-training/w7-iterative-training.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## The Full Post-Training Stack

Putting it all together: pre-training produces a capable but unaligned base model. SFT teaches it to respond in the format and style of a helpful assistant. Reward modeling converts human preference judgments into a differentiable signal. PPO optimizes the policy against that signal while the KL penalty keeps it from going off the rails. DPO offers a cleaner alternative that collapses reward modeling and policy optimization into a single step. Constitutional AI and RLAIF reduce the dependence on expensive human labels. Iterative rounds keep the training signal fresh as the model's distribution evolves.

Each stage exists because the stage before it ran into a wall. SFT hit a ceiling on demonstration quality. RLHF pushed past it but needed a reward model. PPO needed a leash. Reward models got gamed, so you added calibration. Human annotation was too slow, so you added AI feedback. The whole stack is a sequence of answers to problems raised by the previous answer. The diagram below shows how the components connect.

<iframe id="rlhf-w8"
  src="/playgrounds/post-training/w8-full-pipeline_v2.html"
  class="fig-iframe" scrolling="no">
</iframe>

---

## The Core Tension

Every design decision in this stack is, at some level, a response to the same fundamental problem: you want gradient descent to move the model toward "more useful to humans," but you can only give it a concrete, computable loss function. How faithfully that loss reflects what you actually want determines everything.

SFT is limited by the quality of demonstrations. Reward models are limited by the coverage and quality of preference labels. DPO is limited by the diversity of the offline dataset. Constitutional AI is limited by how well the constitution captures human values. Iterative training helps, but the returns diminish. Each technique is a different attempt to approximate what humans value closely enough that optimization actually moves in the right direction.

We're still early. The stack keeps improving, the techniques keep compounding, and the gap between "what the loss measures" and "what we care about" keeps narrowing. Understanding how the pieces fit together isn't just interesting history; it's the prerequisite for pushing any of them further.

---

## Key Papers

<table class="papers-table">
  <thead>
    <tr>
      <th>Year</th>
      <th>Paper</th>
      <th>Contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2017</td>
      <td>Christiano et al., <em>Deep RL from Human Preferences</em></td>
      <td>Original RLHF framework</td>
    </tr>
    <tr>
      <td>2020</td>
      <td>Stiennon et al., <em>Learning to summarize with human feedback</em></td>
      <td>RLHF applied to text generation</td>
    </tr>
    <tr>
      <td>2022</td>
      <td>Ouyang et al., <em>InstructGPT</em></td>
      <td>RLHF at LLM scale</td>
    </tr>
    <tr>
      <td>2022</td>
      <td>Bai et al. (Anthropic), <em>Constitutional AI</em></td>
      <td>CAI and RLAIF</td>
    </tr>
    <tr>
      <td>2017</td>
      <td>Schulman et al., <em>Proximal Policy Optimization</em></td>
      <td>The PPO algorithm</td>
    </tr>
    <tr>
      <td>2023</td>
      <td>Rafailov et al., <em>Direct Preference Optimization</em></td>
      <td>DPO algorithm and theory</td>
    </tr>
    <tr>
      <td>2023</td>
      <td>Lightman et al. (OpenAI), <em>Let's Verify Step by Step</em></td>
      <td>Process reward models</td>
    </tr>
  </tbody>
</table>
