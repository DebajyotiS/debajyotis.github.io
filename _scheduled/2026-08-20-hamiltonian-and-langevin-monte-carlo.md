---
layout: post
title: "The Gradient of the Log-Density Is All You Need to Sample"
date: 2026-08-20
description: Hamiltonian and Langevin Monte Carlo turn the score into a sampler, and MH only has to correct for discretization error, not for using the wrong physics.
categories: GenAI
tags: [ml, generative-modelling, probability, mcmc, langevin]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act II Post 4 of 13.

Outline:
- HMC as momentum augmentation + symplectic integration
- MH step corrects discretization error, not the underlying physics
- NUTS
- Overdamped Langevin, ULA vs MALA
- Fokker-Planck equation as the statement of what the density is doing
- Plant the flag: grad_x log p(x) is sufficient for sampling and doesn't need Z —
  the callback for all of Act IV
-->
