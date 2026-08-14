---
layout: post
title: "You Don't Need to Know a Distribution to Sample From It"
date: 2026-08-19
description: Markov chains and the Metropolis-Hastings ratio let you sample from an unnormalized density, because the normalizing constant cancels.
categories: GenAI
tags: [ml, generative-modelling, probability, mcmc]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act I Post 3 of 13.

Outline:
- Setup shift: unnormalized p-tilde(x), Z unreachable
- Detailed balance, ergodicity
- The MH ratio and why Z cancels
- Gibbs sampling
- Diagnostics done properly: autocorrelation time, ESS, R-hat, why trace plots lie
- Running target: Ising model (pays off later with EBMs in Post 6)
-->
