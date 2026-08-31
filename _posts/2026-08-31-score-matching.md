---
layout: post
title: "Denoising Is Secretly Estimating a Score"
date: 2026-08-31
description: Denoising score matching kills the normalizer via integration by parts, and noise conditioning plus annealed Langevin fixes what naive score matching gets wrong.
categories: GenAI
tags: [ml, generative-modelling, deep-learning, score-matching]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act IV Post 11 of 13. Where Act II and Act
III finally merge.

Outline:
- Hyvarinen's identity, the integration by parts that kills Z
- Sliced score matching
- Denoising score matching (Vincent, 2011) as the pivotal result
- Why naive score matching fails in practice: manifold hypothesis, no signal in
  low-density regions
- The fix: noise conditioning plus annealed Langevin (callback to Post 5)
-->
