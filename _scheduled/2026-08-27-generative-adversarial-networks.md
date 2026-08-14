---
layout: post
title: "GANs Don't Fail to Converge, They Fail to Have an Equilibrium"
date: 2026-08-27
description: Mode collapse and non-convergence are dynamics problems, not optimization problems, and WGAN's Kantorovich duality is a first real contact with optimal transport.
categories: GenAI
tags: [ml, generative-modelling, deep-learning, gan]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act III Post 9 of 13.

Outline:
- Implicit densities, the minimax game
- Optimal-discriminator result and the JS-divergence interpretation
- Mode collapse and non-convergence as dynamics problems, not optimization problems
- WGAN via Kantorovich duality — first real contact with optimal transport, banked
  for rectified flow in Post 13
- Spectral norm, gradient penalty, StyleGAN
- Recent reappraisal (R3GAN): the architectures were fine, the objectives weren't
-->
