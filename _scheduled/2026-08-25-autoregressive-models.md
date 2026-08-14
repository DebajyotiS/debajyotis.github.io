---
layout: post
title: "Autoregressive Models Are the Only Ones Being Honest About the Likelihood"
date: 2026-08-25
description: The chain rule gives exact tractable likelihood and O(d) sampling, at the cost of committing to an ordering that isn't there in the data.
categories: GenAI
tags: [ml, generative-modelling, deep-learning, autoregressive]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act III Post 7 of 13.

Outline:
- Chain rule decomposition, exact likelihood, O(d) sampling
- PixelCNN -> WaveNet -> transformers
- Teacher forcing, exposure bias
- The arbitrariness of the ordering
- Why this earns its own post: only family with exact tractable likelihood, and
  discrete diffusion is currently trying to eat its lunch
-->
