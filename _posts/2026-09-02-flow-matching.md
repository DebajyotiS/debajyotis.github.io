---
layout: post
title: "Flow Matching Deletes the Simulation, Not the Idea"
date: 2026-09-02
description: The conditional flow matching trick replaces an intractable marginal velocity field with a closed-form conditional one that shares its gradient, keeping continuous transport without the ODE-in-the-loop cost.
categories: GenAI
tags: [ml, generative-modelling, deep-learning, flow-matching]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act IV Post 13 of 13.

Outline:
- Conditional flow matching trick: marginal velocity field is intractable, the
  conditional one is closed-form, and the two objectives share a gradient
- OT paths, rectified flow, straightness and why it buys few-step sampling
- Stochastic interpolants as the umbrella containing diffusion as a special case
  of schedule choice
- Reflow, consistency models, distillation
- Optional follow-on: Post 14 synthesis/epilogue (comparison table across
  path/parameterization/objective/sampler, evaluation, discrete data, open problems)
-->
