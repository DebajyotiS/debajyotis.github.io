---
layout: post
title: "Diffusion's Noise Schedule Is Older Than Diffusion"
date: 2026-08-21
description: Parallel tempering, annealed importance sampling, and SMC are the direct ancestors of the noise schedules that show up later in score-based generative models.
categories: GenAI
tags: [ml, generative-modelling, probability, mcmc, smc]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act II Post 5 of 13. (Marked optional/mergeable
in the plan — could fold into a sidebar of Post 11 if the series gets compressed.)

Outline:
- Parallel tempering
- Annealed importance sampling (AIS)
- Sequential Monte Carlo (SMC)
- Thermodynamic integration
- Why this matters: direct ancestor of annealed Langevin sampling in NCSN, and the
  reason diffusion's noise schedule is an inheritance, not a trick
-->
