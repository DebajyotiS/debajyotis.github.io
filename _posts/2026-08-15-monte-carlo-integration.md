---
layout: post
title: "How Do We Sample Anything at All? (Part 3: What Can We Actually Do With Samples?)"
date: 2026-08-15
description: Turning integration into simple averaging using Monte Carlo integration, importance sampling, sequential methods, and variance reduction.
categories: GenAI
tags: [ml, generative-modelling, monte-carlo, bayesian, probability]
featured: false
giscus_comments: false
related_posts: true
---

In [Part 1]({% post_url 2026-08-13-inverse-transform-sampling %}), we learned how to transform uniform random variables into draws from known probability distributions. In [Part 2]({% post_url 2026-08-14-mcmc-sampling %}), we tackled intractable targets using Markov Chain Monte Carlo and Hamiltonian dynamics. 

Across both posts, we treated getting samples as the end goal. But collecting thousands of numbers in a vector is not inherently useful by itself. It is time to ask a practical question: **what are those samples actually good for?**

The short answer is surprising: **sampling is the ultimate tool for integration.**

## The Problem: Intractable Expected Values

In modern machine learning, statistics, and physics, we constantly run into integrals that look like this:

$$I = \mathbb{E}_{p}[f(X)] = \int f(x) p(x) \, \mathrm{d}x$$

If $x$ is a single continuous variable, you might solve this integral with calculus or basic numerical quadrature. But if $x$ represents an image with thousands of pixels, a set of parameters in a neural network, or a complex physical system, analytical integration fails completely. High-dimensional space is far too vast to evaluate on a grid.

The fundamental trick of Monte Carlo estimation is turning that impossible calculus problem into a simple arithmetic average:

$$\hat{\mu}_N = \frac{1}{N} \sum_{i=1}^{N} f(x_i), \qquad x_i \sim p(x)$$

Instead of calculating the area under a curve analytically, you draw $N$ independent samples from $p(x)$, pass each sample through $f(x)$, and calculate their mean. 

## A Concrete Example: Integrating Without Integration

To see this in action, suppose you want to compute a simple definite integral:

$$I = \int_{0}^{1} x^2 \, \mathrm{d}x$$

Standard calculus tells us the exact answer is $\frac{1}{3} \approx 0.3333$. 

To solve this via Monte Carlo, notice that the integrand $x^2$ is multiplied by the uniform density $p(x) = 1$ on the interval $[0,1]$. You generate $N$ uniform random numbers $x_1, \ldots, x_N \sim \mathcal{U}(0,1)$, square every single one, and average the results:

```python
import numpy as np

# Generate 1,000,000 uniform random numbers
samples = np.random.uniform(0.0, 1.0, size=1000000)

# Evaluate f(x) = x^2 and compute the mean
mc_estimate = np.mean(samples ** 2)
# Result: ~0.3333

```

Without doing any calculus, the sample average converges directly onto the exact integral. Sampling turns integration into counting.

## Why Does Averaging Work? (The Convergence Rules)

Two bedrock theorems of probability theory guarantee that this simple counting trick works.

First, the **Law of Large Numbers (LLN)** states that as your sample size $N$ grows toward infinity, the sample average converges almost surely to the true expected value:

$$\frac{1}{N} \sum_{i=1}^{N} f(x_i) \xrightarrow{\text{a.s.}} \mathbb{E}[f(X)]$$

Second, the **Central Limit Theorem (CLT)** tells us how fast that convergence happens. For large values of $N$, the distribution of our estimator $\hat{\mu}_N$ approaches a Gaussian centered on the true mean $\mu$:

$$\hat{\mu}_N \approx \mathcal{N}\left(\mu, \frac{\sigma_f^2}{N}\right)$$

The standard error of our Monte Carlo estimate scale according to a famous relationship:

$$\boxed{\text{Standard Error} \propto \frac{1}{\sqrt{N}}}$$

This square-root relationship carries a massive practical catch. **To reduce your estimation error by a factor of 10, you must generate 100 times as many samples.**

```
Sample Size (N)     Standard Error Reduction
───────────────     ────────────────────────
            100     Baseline Error
         10,000     1/10th Error      (100x samples required)
      1,000,000     1/100th Error    (10,000x samples required)

```

While a $\frac{1}{\sqrt{N}}$ convergence rate seems slow, Monte Carlo estimation possesses a superpower that deterministic grid integration lacks: **its convergence rate does not depend on the dimensionality of $x$.**

| Method | Convergence Error in $d$ Dimensions | Cost for High Dimensions |
| --- | --- | --- |
| **Grid Integration** | $\mathcal{O}(N^{-1/d})$ | Exponential Explosion (Curse of Dimensionality) |
| **Monte Carlo Integration** | $\mathcal{O}(N^{-1/2})$ | Completely Independent of Dimension $d$ |

Evaluating a 100-dimensional integral with a grid of 10 points per axis requires $10^{100}$ evaluations, which exceeds the number of atoms in the observable universe. Monte Carlo estimation handles 100 dimensions with the exact same mathematical convergence rate as a 1-dimensional line.

## Importance Sampling: Spending Samples Where They Matter

Sometimes $p(x)$ is easy to evaluate but hard to sample from directly, or $f(x)$ is nearly zero almost everywhere except for a tiny region that dominates the integral.

We can rewrite our expectation using a change of measure under an alternative proposal distribution $q(x)$:

$$\mathbb{E}_p[f(X)] = \int f(x) p(x) \, \mathrm{d}x = \int f(x) \frac{p(x)}{q(x)} q(x) \, \mathrm{d}x = \mathbb{E}_q\left[ f(X) \frac{p(X)}{q(X)} \right]$$

This leads to the **Importance Sampling Estimator**:

$$\hat{\mu}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^{N} f(x_i) w(x_i), \qquad x_i \sim q(x)$$

The quantity $w(x) = \frac{p(x)}{q(x)}$ acts as an importance weight that corrects for sampling from $q(x)$ instead of $p(x)$.

{% include figure.liquid path="assets/img/sampling-series/importance-sampling.svg" class="img-fluid rounded" alt="Target density p(x) and proposal density q(x), with the region where q's thinner tail inflates the importance weight shaded." caption="Sampling from the easy q(x) instead of the hard p(x) is fine everywhere the two densities track each other — the danger is q's thinner tail, where a rare sample gets an enormous corrective weight." %}

Importance sampling allows you to focus your computation on the regions that matter most.

However, importance sampling has a dangerous failure mode known as **weight degeneracy**. If your proposal distribution $q(x)$ has "lighter tails" than $p(x)$, there will be regions where $p(x)$ is non-zero but $q(x)$ is near zero. Samples landing in those regions will receive immense importance weights, causing your estimator's variance to explode.

## Sequential Monte Carlo: Particle Filters

If a single importance sampling proposal $q(x)$ struggles to cover a target distribution over time, we can extend the idea into a sequence of evolving distributions:

$$p_0(x) \longrightarrow p_1(x) \longrightarrow \cdots \longrightarrow p_T(x)$$

This is the foundation of **Sequential Monte Carlo (SMC)**, also known as particle filtering.

Instead of drawing one sample and keeping it forever, SMC maintains a cloud of weighted points called particles. As time steps forward or data arrives:

1. **Mutate:** Particles are propagated forward using a proposal mechanism.
2. **Reweight:** Each particle's weight is updated using the likelihood of new incoming observations.
3. **Resample:** Particles with low weights are eliminated, while high-weight particles are duplicated.

This survival-of-the-fittest cycle prevents weight degeneracy, ensuring your computational budget stays focused on high-probability trajectories.

## MCMC as Monte Carlo With Correlated Samples

In Part 2, we saw that MCMC generates sequences of correlated samples $x_1, x_2, \ldots, x_N$. Fortunately, the Monte Carlo sample mean still converges to the correct expectation even when samples are correlated:

$$\hat{\mu}_{\text{MCMC}} = \frac{1}{N} \sum_{i=1}^{N} f(x_i) \xrightarrow{\text{a.s.}} \mathbb{E}_p[f(X)]$$

However, correlation increases the variance of our estimator. We measure this correlation through the autocorrelation function $\rho_k = \text{Corr}(f(x_t), f(x_{t+k}))$, which quantifies how much a sample at step $t$ tells us about the sample $k$ steps later.

To account for this redundancy, we calculate the **Effective Sample Size (ESS)**:

$$N_{\text{eff}} = \frac{N}{1 + 2 \sum_{k=1}^{\infty} \rho_k}$$

If your MCMC chain has an $N_{\text{eff}}$ of 1,000 out of 50,000 total steps, your correlated chain contains the exact same statistical power as 1,000 purely independent draws.

## Quantifying Uncertainty With Bayesian Posteriors

In Bayesian statistics, Monte Carlo sampling provides far more than point estimates. Once you have a set of samples from a posterior distribution $\theta^{(1)}, \ldots, \theta^{(N)} \sim p(\theta \mid y)$, you can extract any statistical property you need without evaluating complex formulas:

* **Posterior Mean:** $\mathbb{E}[\theta \mid y] \approx \frac{1}{N} \sum_{i=1}^{N} \theta^{(i)}$
* **Posterior Variance:** $\text{Var}(\theta \mid y) \approx \frac{1}{N} \sum_{i=1}^{N} \left(\theta^{(i)} - \hat{\mu}\right)^2$
* **Credible Intervals:** Sort your samples and find the 2.5th and 97.5th percentiles directly.
* **Complex Transformations:** To find the distribution of some arbitrary function $g(\theta)$, simply transform each sample $g(\theta^{(i)})$ and plot the resulting histogram.

{% include figure.liquid path="assets/img/sampling-series/posterior-credible-interval.svg" class="img-fluid rounded" alt="Posterior density p(theta | y) with the median marked and the 95% credible interval shaded between the 2.5th and 97.5th percentiles." caption="Every quantity here — the median, the shaded interval — is read straight off the sorted samples, no closed-form posterior required." %}

You do not need to derive closed-form distributions for transformed variables. The empirical distribution of the samples handles the algebra automatically.

## Flipping the Script: The Bridge to Generative AI

Across this three-part series, we have moved in one consistent direction: start from a known distribution $p(x)$, generate samples $x_i$, then use those samples to solve computations. We assumed from the beginning that someone handed us an analytical expression for $p(x)$, and used inverse CDFs, rejection envelopes, MCMC chains, and Monte Carlo estimators to pull samples out of that known distribution.

Real data rarely comes with that luxury attached. A giant folder of images, a dataset of audio clips, a repository of text documents — in each case you possess the samples $x_1, x_2, \ldots, x_N \sim p_{\text{data}}(x)$, but $p_{\text{data}}(x)$ itself is an unknown, high-dimensional probability density that nobody wrote down for you.

The central problem of modern Generative AI is reversing our entire pipeline:

$$\boxed{
\text{Observed Samples } x_i \quad \longrightarrow \quad \text{Learn Distribution } p_\theta(x) \quad \longrightarrow \quad \text{Sample New Data } z \sim p(z)
}$$

Instead of deriving a sampler from a known distribution, we build neural networks that learn the underlying distribution directly from raw data.

In our upcoming series on generative modeling, we will see how Variational Autoencoders (VAEs), Normalizing Flows, Diffusion Models, and Flow Matching solve this problem, turning the sampling foundations we built here into tools that can generate synthetic reality.
