---
layout: post
title: "One Uniform Distribution Is Enough, Until the Dimension Grows"
date: 2026-08-18
description: Inverse-CDF, rejection sampling, and importance sampling all turn uniform bits into arbitrary distributions in 1D — and all three die exponentially in high dimension.
categories: GenAI
tags: [ml, generative-modelling, probability, sampling]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act I Post 2 of 13.

Outline:
- Inverse-CDF method
- Box-Muller and Marsaglia polar
- The Ziggurat algorithm
- Rejection sampling, acceptance rate 1/M
- Importance sampling, self-normalized IS, effective sample size
- Punchline / motivating crisis for Act II: acceptance rates and IS variance both
  collapse exponentially with dimension
- Optional: QMC / low-discrepancy sequences as a teaser
-->
