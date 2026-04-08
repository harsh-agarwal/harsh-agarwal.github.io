---
layout: playground-post
title: "Why Muon Wins: An Interactive Demo"
date: 2026-04-06
playground_script: /playgrounds/muon-demo.js
---

Muon is a recently proposed optimizer for neural networks that orthogonalizes gradients using Newton-Schulz iterations before applying updates. The intuition: instead of following the raw gradient (which is dominated by the largest singular directions), Muon normalizes all singular values to 1 — giving equal update strength to every direction in weight space.

The demo below lets you watch four optimizers — SGD, Adam, AdamW, and Muon — compete on a simple matrix associative memory task. An 8×8 weight matrix **W** must learn to map 8 key vectors to value vectors (think: a tiny transformer attention layer storing facts). The catch: training data follows a Zipf distribution. "Cat" appears ~45% of the time; "quokka" appears less than 1%.

This imbalance creates a skewed gradient dominated by a few large singular values. SGD and Adam chase that skew — they learn common pairs well but fail on rare ones. Muon's orthogonalization flattens the singular spectrum, so rare pairs get just as much gradient signal as common ones.

Hit **Run** and watch the per-pair error bars at the bottom. The rare pairs (quokka, sushi, mars) are where Muon's advantage shows most clearly.

<div id="playground-muon-root" style="margin: 2em 0;"></div>
