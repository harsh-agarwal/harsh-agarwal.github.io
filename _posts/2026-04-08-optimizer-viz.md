---
layout: post
title: "Alec Radford Kind of Visualization of Muon"
date: 2026-04-08
---

One of the clearest ways to build intuition for optimizers is Alec Radford's classic contour visualization — watch each optimizer trace a path across a loss landscape in real time. SGD with momentum hugs ravines slowly. Adam cuts through efficiently. Muon, with its orthogonalized gradient updates, takes a distinctly different trajectory.

The visualization below runs four optimizers simultaneously on classic 2D test functions: Rosenbrock, Beale, Himmelblau, and Ackley. You can toggle optimizers on/off, change learning rates independently, switch surfaces, and even click to set a custom starting point.

<iframe
  src="/playgrounds/optimizer-viz.html"
  style="width:100%; height:860px; border:none; border-radius:8px; margin: 1.5em 0; display:block;"
  loading="lazy"
  title="Optimizer comparison visualization"
></iframe>

A few things worth noticing:

- **Rosenbrock** (the banana function) is the classic test — the global minimum is at (1,1) inside a long narrow curved valley. SGD overshoots and oscillates; Muon's direction-normalized updates track the valley more cleanly.
- **Himmelblau** has four equal global minima — watch which one each optimizer finds depending on the starting point.
- **Same LR mode** is the fairest comparison. Enable it and see how much Muon's structural advantage holds when all optimizers use identical learning rates.
