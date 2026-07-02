---
layout: playground-post
title: "Harness Engineering: From Wire Bundles to AI Agents"
date: 2026-07-02
description: "Why software engineers borrowed a term from aerospace wiring to describe the discipline of building AI agents — and what the physical world's hard-won lessons about reliability have to say about it."
tags: [ai-agents, engineering, llm, machine-learning]
---

Earlier this year, OpenAI shipped a production application at significant scale, and the team behind it didn't write a single line of the application code. The model wrote it all. What the team *did* build was the scaffolding around the model: the system prompt, the tool definitions, the execution loop, the memory management, the verification checks, the guardrails. They called that scaffolding the **harness**. And the discipline of designing it, harness engineering, is quietly becoming the most important skill in AI agent development.

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
</style>

I've been thinking about why this particular word stuck. It didn't come from nowhere. "Harness" has been an engineering term for over a century, and the physical version of the concept turns out to be surprisingly instructive for understanding the AI version. So this post does something a little unusual: it traces the idea from the physical world into machine learning, and explains why the metaphor isn't just cute. It's structurally accurate.

---

## The Core Idea: Agent = Model + Harness

Here's the formulation that's been making the rounds: **Agent = Model + Harness**. Everything that isn't the model (the system prompt, the tool definitions, the memory architecture, the execution loop, the error handling, the guardrails) is the harness.

This matters because models are becoming commodity. GPT-4o, Claude, Gemini: at the agent layer, they're increasingly interchangeable reasoning engines. The thing that makes an AI agent useful, reliable, and safe isn't which model you pick. It's the harness you wrap around it.

A model without a harness is a reasoning engine with no steering wheel. It can think, but it doesn't know what tools it has, can't manage its own memory across steps, doesn't know when to stop, and has no way to recover from errors. The harness is what gives it all of that.

Martin Fowler's team formalized much of the vocabulary here (guides, sensors, steering loops) and it's becoming standard. But the underlying insight is simple: the model is the engine; the harness is everything else in the vehicle.

---

## What a Harness Actually Contains

Let me be concrete about what lives inside this layer.

**The system prompt** is the most basic element. It tells the model who it is, what tools it has, how it should behave, and what it must not do. Static, feedforward, always present.

**Tool definitions** give the model hands. Without them, a model can only produce text. With them, it can call a search API, write to a file, run terminal commands, query a database. The harness defines which tools exist and what calling them means.

**The execution loop** drives the model through multi-step tasks. At each step: observe the state, call the model, execute any tool calls, feed results back, repeat. This loop also decides when the task is done, when to escalate to a human, and when to abort. Think of it as the heartbeat of the harness, the component that keeps everything turning in one direction. (I'll come back to this analogy.)

**Memory and context management** is probably the hardest problem. Models have finite context windows. A harness for a long-running task must decide what to keep, what to compress, and what to retrieve from external storage. Get this wrong and the model loses the thread of what it's doing mid-task.

**Sensors and verification** are the feedback side. Computational sensors (linters, type checkers, test runners) are cheap and fast. Inferential sensors (AI-based code review, semantic checks) are slower and more expensive but catch issues that deterministic tools miss. A good harness uses both, feeding their outputs back to the model in a form it can act on.

**Guardrails** define what the model must not do. Hard stops. These can be prompt-level instructions, runtime checks that intercept tool calls before execution, or post-processing filters on outputs.

---

## Where the Word Comes From

The word "harness" didn't appear in ML from thin air. It traveled through three distinct stops, and the journey is worth tracing because it explains why the metaphor works so well.

**The physical harness** is where it all starts. The word comes from Old French *harneis*, meaning equipment or armor, and by the 14th century it named the collar, traces, and straps that connect a draft horse to a cart. The tack doesn't make the horse stronger. It doesn't compete with the horse. It channels the horse's power into directed, useful work. That's the original meaning, and it's exactly the meaning the AI usage inherits.

In electrical engineering, a **wiring harness** is a structured assembly of conductors (wires, cables, connectors) bound together in a defined geometry and routed through a machine. A modern car has 1,500 to 3,000 meters of wiring. A commercial airliner has hundreds of miles. A fighter jet packs enormous electrical capability into a tight airframe while surviving G-forces, temperature extremes, and battle damage. In aerospace, every wire is gauged, every connector MIL-SPEC rated, every route designed to separate power from signal, and every junction tested against failure modes that have caused real accidents. The harness is what makes the physics reliable under load.

In software, the **test harness** is the scaffolding around a unit under test: setup code, mock dependencies, assertion framework. It doesn't test anything itself; it creates the controlled environment in which the test can run. Software engineers have used this term for decades.

Then came the **evaluation harness** in ML. As teams needed to benchmark models against standard tasks, they built suites that ran a model through hundreds of scenarios and recorded performance. SWE-bench calls its execution scaffold a *harness*. This is what established the term in ML: the harness as scaffolding that puts a model through its paces.

The **agent harness** is the newest arrival, and it's where the term is currently exploding. When models stopped being oracles (ask a question, get an answer) and started being agents (write code, browse the web, modify repositories), someone had to build the structure that gave them tools, memory, direction, and guardrails. That structure inherited the name.

---

## A Principle Is Not a Harness

There's a distinction the physical world makes cleanly that I think the software world often blurs: the difference between a *principle* and a *harness*.

Take Fleming's Left Hand Rule. Point your left hand with your index finger along the magnetic field, your middle finger in the direction of current flow, and your thumb points in the direction of the resulting force. That's the Lorentz force law in three fingers, the physical principle behind every DC motor ever built.

But the principle is not the motor.

A DC motor wraps that principle in an engineered system: a wound armature, permanent magnets, brushes, a housing, a shaft, and critically, a commutator. The commutator is the interesting part. It reverses current through each armature coil at exactly the moment when doing so keeps torque pushing in one direction. Without it, you'd get a twitch, not continuous rotation.

Fleming's rule tells you *what will happen* when current flows through a field. The motor harness makes *useful rotation happen reliably*, under load, across temperature ranges, for years. The principle is the "why." The harness is the "how, controllably."

This maps directly to ML. The transformer architecture with its attention mechanism is the principle, the "why it works." The agent harness (system prompt, tools, execution loop, memory, guardrails) is the engineered system that makes the principle produce useful, directed output.

And that execution loop I mentioned earlier? It's the commutator. It's the mechanism that keeps things turning in one direction. At each step (call model, execute tools, feed results back, decide whether to continue) it prevents the system from oscillating and ensures the principle produces continuous, directed work.

<iframe src="/playgrounds/harness-engineering/visual-1-principle-vs-harness.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="Visual: Principle vs Harness — Fleming's Rule and the DC Motor"></iframe>
<p class="fig-note">Step through: (1) Fleming's Left Hand Rule, (2) the DC motor as harness, (3) the same structure in ML.</p>

---

## Leverage vs. Harness

There's a related concept that keeps getting conflated with harness engineering: *leverage*.

Archimedes' lever amplifies force. A small effort at the long end produces a large force at the short end. In AI, the leverage claim is real. A single well-crafted prompt can produce pages of code or hours of analysis. The model is the fulcrum, and you're pushing at the long end of an extremely powerful lever.

But leverage doesn't tell you which direction the force goes.

A lever amplifies equally in all directions. It has no concept of boundary, goal, or stop condition. It is pure force multiplication. A harness is different. The horse collar and traces of a draft harness don't make the horse stronger. They ensure the horse's strength moves the cart in the intended direction, at a controllable pace, with the ability to stop.

In ML: the model provides leverage (amplification of reasoning). The harness provides direction (tools, constraints, memory, feedback loops). You need both. A powerful model without a harness is ungovernable. A harness around a weak model is just scaffolding around a brick.

The reason "harness engineering" is winning as a discipline name over "leverage engineering" is telling. Leverage is about what you *observe*. Harness is about what you *build*. One is a property; the other is a discipline. Engineers build things.

<iframe src="/playgrounds/harness-engineering/visual-2-leverage-vs-harness.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="Visual: Leverage vs Harness"></iframe>
<p class="fig-note">Four steps: (1) what a lever does, (2) what a harness does, (3) the explicit contrast, (4) both concepts in an LLM agent.</p>

---

## Harness Engineering at Different Levels

The discipline scales, just like it does in the physical world, from a garden tractor to a spacecraft.

**At the tool level**, think Claude Code or GitHub Copilot. The harness is pre-built. The product ships with a system prompt, a set of tools, and an execution loop. You can extend it with project-specific rules, but the structure is there. Consumer tier.

**At the application level**, a custom AI agent for an enterprise workflow, the harness *is* the engineering deliverable. The team picks a model, wraps it with domain-specific tools (CRM APIs, internal databases, approval workflows), designs the memory architecture, writes the guardrails, builds the verification sensors. The model is commodity here. The harness is the moat.

**At the evaluation level**, benchmark suites like SWE-bench, the harness defines what "performance" means. Which tasks, what execution environment, what constitutes a correct answer, how failures are categorized. A poorly designed eval harness produces misleading benchmarks. A well-designed one becomes the industry standard.

**At the training level**, RLHF and constitutional AI, the harness structures how model outputs are evaluated and how those evaluations feed back into weight updates. This is the highest-stakes tier. Get the harness wrong and you encode the error into the model itself.

---

## The Open Problems

This is a young discipline. The hard problems are unsolved.

**Silent failure.** A physical wiring harness fails in predictable ways: a wire corrodes, a connector loosens. An AI harness can fail silently. The model appears to succeed while producing subtly wrong outputs. Distribution shift makes this worse: a harness built for one class of tasks can behave unpredictably when inputs drift.

**Vocabulary confusion.** "Harness" currently means the whole product, the orchestration layer, an SDK, an IDE plugin, and an eval scaffold, sometimes in the same conversation. This polysemy blocks meaningful comparison and accumulation of design knowledge. The field needs sharper terms.

**Verification depth.** Linters can check syntax. Type checkers can check structure. Nothing cheap can check intent. When an agent modifies a production system, the question isn't whether the code compiles, it's whether it does what was wanted. Semantic verification is expensive, non-deterministic, and still an open research problem.

**When to involve a human.** The harness engineer's job is to improve guides and sensors whenever an issue recurs, to automate the judgment and push the human up the stack. But as agents get more capable and tasks get more complex, deciding *when* a human should still be in the loop is increasingly non-obvious. There's no formula for this yet.

---

## The Pattern Underneath

I keep coming back to the same observation: the harness is always the middle layer. Between the physics textbook and the product spec. Between the principle that makes something possible and the output that makes it useful. You can't point at it in the textbook. You can't point at it in the product. It lives in the engineering work between the two.

The physical world learned this over a century of hard failures. Chafed wires in jet engines. Corroded pins in spacecraft. The ML world is a few years in. The failures are less visible but not less real.

The best wire in the world, routed carelessly, will fail. The best language model, wrapped in a careless harness, will too. The word traveled across domains because the engineering challenge is the same.

<iframe src="/playgrounds/harness-engineering/visual-3-layer-stack.html"
        class="fig-iframe" scrolling="no"
        onload="this.style.height=(this.contentWindow.document.body.scrollHeight+2)+'px'"
        title="Visual: The Harness Layer Stack"></iframe>
<p class="fig-note">Three steps: (1) the universal four-layer pattern, (2) the same stack across DC motors, aircraft wiring, and LLM agents, (3) the key insight distilled.</p>

---

*Further reading: [SWE-bench](https://www.swebench.com/) for the eval harness that benchmarks coding agents · [Martin Fowler on harness engineering](https://martinfowler.com/articles/harness-engineering.html) · AS50881 (aerospace wiring standard) · MIL-DTL-22759 (military wire specification)*
