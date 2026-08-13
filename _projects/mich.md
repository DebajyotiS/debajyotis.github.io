---
layout: page
title: MICH
description: Physics-informed neural network that inverts layer-resolved laminar fMRI BOLD signals to recover the underlying neural activity.
importance: 2
category: work
github: https://github.com/DebajyotiS/mich
redirect: https://github.com/DebajyotiS/mich
---

MICH (Machine Inference for Cortical Haemodynamics) tackles an ill-posed inverse problem: laminar fMRI measures BOLD signals separately across cortical layers, an indirect and noisy readout of the underlying neural activity shaped by haemodynamic coupling (the Balloon model), point-spread contamination across layers, and acquisition noise.

The project inverts that process with a physics-informed neural network (PINN) that jointly fits the observed data and penalises violations of the Balloon model ODEs, using a collocation-based physics loss over continuous space and time.

[View the repository on GitHub]({{ page.github }})
