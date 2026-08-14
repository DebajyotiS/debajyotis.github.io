---
layout: post
title: "The Reparameterization Trick Is Just Inverse Transform Sampling Again"
date: 2026-08-26
description: The ELBO decomposition makes the approximation gap interpretable, and a very deep hierarchical VAE with a fixed inference path is nearly a diffusion model.
categories: GenAI
tags: [ml, generative-modelling, deep-learning, vae]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act III Post 8 of 13.

Outline:
- Derive ELBO via log p(x) = ELBO + KL(q||p), not Jensen, so the gap is interpretable
- Amortized inference
- Reparameterization trick — flag explicitly as Post 2's transform method reappearing
- Posterior collapse, why Gaussian likelihoods produce blur
- Beta-VAE, VQ-VAE
- End on hierarchical VAEs (NVAE, VDVAE): a very deep hierarchy with a fixed
  inference path is nearly a diffusion model — cliffhanger into Act IV
-->
