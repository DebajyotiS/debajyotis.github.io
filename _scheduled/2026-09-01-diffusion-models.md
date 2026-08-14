---
layout: post
title: "Diffusion Models Are Just Score Matching with Better Bookkeeping"
date: 2026-09-01
description: DDPM's variational bound collapses into a score-prediction MSE, and the probability-flow ODE gives identical marginals to the reverse-time SDE with deterministic dynamics.
categories: GenAI
tags: [ml, generative-modelling, deep-learning, diffusion]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act IV Post 12 of 13. Likely the longest;
the one place a split is tolerable.

Outline:
- DDPM: forward process, closed-form q(x_t | x_0), variational bound collapsing
  into epsilon-prediction MSE
- Equivalence of the epsilon / x_0 / score / v parameterizations
- Continuous view: VP and VE SDEs, Anderson's reverse-time SDE
- Probability-flow ODE: identical marginals, deterministic dynamics
- Guidance: classifier guidance and CFG
- Samplers: DDIM, DPM-Solver, Heun/EDM, Karras's design-space reframing
- Latent diffusion
-->
