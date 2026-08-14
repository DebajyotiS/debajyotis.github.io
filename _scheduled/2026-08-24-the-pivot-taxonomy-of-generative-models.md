---
layout: post
title: "Every Generative Model Is Dodging the Same Intractable Integral"
date: 2026-08-24
description: Once you only have samples and not a density, MLE becomes forward-KL minimization, and every model family is defined by how it evades the normalizer or the marginal.
categories: GenAI
tags: [ml, generative-modelling, deep-learning]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act III Post 6 of 13. The pivot post.

Outline:
- No more p-tilde(x); only samples
- MLE as forward-KL minimization
- Taxonomy by how each family evades the intractable normalizer/marginal:
  - EBMs need MCMC in the training loop (contrastive divergence — direct callback
    to Acts I/II)
  - Latent-variable models have an intractable marginal integral p(x|z)p(z) dz
  - Three escapes: constrain the architecture for exact likelihood / bound the
    likelihood / abandon likelihood entirely
- Introduce the pushforward / change-of-variables view — load-bearing for the rest
  of the series
-->
