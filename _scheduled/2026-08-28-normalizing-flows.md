---
layout: post
title: "Normalizing Flows Trade Sampling Speed for Density, or Vice Versa"
date: 2026-08-28
description: Coupling and autoregressive flows force a choice between fast density and fast sampling, and continuous flows trade that for an ODE solve inside the training loop.
categories: GenAI
tags: [ml, generative-modelling, deep-learning, normalizing-flows]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---

<!--
Series: Sampling / generative modeling, Act IV Post 10 of 13.

Outline:
- Change of variables, the log-det bottleneck
- Coupling layers (NICE, RealNVP)
- Autoregressive flows, the MAF/IAF duality: fast density xor fast sampling, pick one
- Splines, Glow
- Continuous normalizing flows: neural ODEs, instantaneous change-of-variables,
  Hutchinson trace estimation, FFJORD
- End on the cost problem — ODE solves inside the training loop — that flow
  matching (Post 13) deletes
-->
