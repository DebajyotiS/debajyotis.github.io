---
layout: post
title: "How Do We Sample Anything at All? (Part 1: The Inversion Trick)"
date: 2026-08-13
description: A friendly exploration of how deterministic computers generate probability distributions from scratch using inverse transform sampling.
categories: GenAI
tags: [ml, generative-modelling, probability]
featured: false
giscus_comments: false
related_posts: true
---

There is an unassuming snippet of code that quietly sneaks its way into almost every machine learning repository on the planet:

```python
import numpy as np

rng = np.random.default_rng(42)
x = rng.random()
```

Just like that, we have a random number. But if we pause to ask a slightly annoying question, the magic starts to unravel: **where on earth did that number actually come from?**

Computers are fundamentally deterministic machines. Hand a processor the exact same inputs alongside the exact same internal state, and it will give you the exact same outputs every single time. So how does a completely deterministic box of silicon produce something we can honestly call "random"?

Even if we solve that puzzle, a bigger one is immediately waiting for us in the wings. Suppose you do not want a simple decimal between 0 and 1. What if you need to draw samples from a Gaussian curve, an exponential decay, or some absurd custom distribution you just scribbled on a napkin? Your random number generator only gives you uniform floats between 0 and 1. How do you bend, stretch, and fold those plain decimals until they look like samples from the distribution you actually care about?

In this post, we are going to answer that question from first principles. We will start with essentially nothing: a deterministic computer and a humble stream of random bits, building up to a general recipe for transforming simple random numbers into samples from known probability distributions.

## Computers Are Secretly Predictable

Let's begin with an uncomfortable truth: computers do not naturally produce randomness. A standard program executes instructions with absolute determinism. If you evaluate `x = f(42)`, assuming `f` has no sneaky side effects or external network reads, you will get the exact same `x` today, tomorrow, and heat-death-of-the-universe years from now.

When NumPy effortlessly hands you a decimal via `rng.random()`, there isn't a tiny atmospheric noise detector sitting on your CPU. It is running an algorithm. In fact, if you re-initialize the generator with `np.random.default_rng(42)` and call `rng.random()` again, you will get the exact same value down to the last digit.

```python
# Run this as many times as you like, the output never changes
rng = np.random.default_rng(42)
print(rng.random())  # Always 0.7739560485559809
```

This is why we call these systems **pseudo-random number generators** (PRNGs). They take a starting point, known as a seed, and deterministically compute a sequence of numbers that mimics the statistical behavior of pure noise. For almost every practical application, we do not need fundamental quantum randomness; we just need numbers that pass statistical tests for independence and uniformity.

Far from being a drawback, this predictability is actually a superpower in machine learning and data science. Setting a seed ensures that your experiment is completely reproducible. If your model achieves brilliant results with seed 42, anyone else on earth can run your code and verify the exact same results. Reproducibility beats philosophical purity every single time.

## Asking the Right Thing from a Random Generator

Before tackling complex distributions, let's look at what the basic random number generator gives us out of the box: a single number picked uniformly between 0 and 1.

Mathematically, we write this as $U \sim \mathcal{U}(0,1)$. The notation looks fancy, but the underlying concept is straightforward. $U$ is a random variable where every equal-width sub-interval inside $[0,1]$ has an identical chance of receiving a point. For instance, the probability of a draw landing between 0.0 and 0.1 is exactly 10%, the chance of landing between 0.4 and 0.5 is 10%, and the chance of landing between 0.9 and 1.0 is also 10%.

$$P(0 \leq U < 0.1) = P(0.4 \leq U < 0.5) = P(0.9 \leq U \leq 1.0) = 0.1$$

If you sample a million points from this generator and plot them as a histogram, you will see a flat, featureless block:

```python
samples = rng.random(1_000_000)
```

Individual draws are unpredictable, but en masse, they reveal a perfectly flat structure. This distinction highlights something crucial about probability: a random variable isn't interesting because an individual point feels unpredictable; it is interesting because **the overall collection of samples obeys a specific shape**.

Which brings us back to our core problem: what if we need a shape that isn't flat?

## Finding a Ruler for Probability

Suppose someone hands you a target probability distribution $p(x)$ that rises in some places and falls in others. There are regions where points should cluster densely and quiet valleys where points rarely land. To draw samples correctly, your output values must land in those peaks and valleys with precise mathematical probabilities.

To do that, we need a way to describe those probabilities quantitatively. The most common tool is the **probability density function** (PDF), written as $p(x)$. For continuous variables, the PDF describes how probability mass is spread across the real line, meaning the probability of a sample falling within a tiny window around $x$ is approximately $p(x)\mathrm{d}x$.

While the PDF gives a great visual profile of a distribution, a different representation turns out to be vastly more useful for sampling: the **cumulative distribution function** (CDF), defined as $F_X(x) = P(X \leq x)$.

$$F_X(x) = P(X \leq x)$$

You can think of the CDF as sliding a vertical bar across your distribution from left to right, sweeping up probability mass as it goes. On the far left, you have gathered nothing, so $F_X(x) \approx 0$. By the time your sliding bar reaches the far right, you have swept up all the probability, making $F_X(x) \approx 1$.

Notice what just happened. The CDF takes any physical value $x$ from your distribution and maps it to an output number between 0 and 1. Meanwhile, our basic computer random generator spits out values between 0 and 1. That structural symmetry is not a coincidence; it is the master key to sampling.

## Running the Machine in Reverse

Let's revisit the standard uniform distribution $U \sim \mathcal{U}(0,1)$. Because probability accumulates at a constant rate across the unit interval, its cumulative distribution function is astonishingly simple: $F_U(u) = u$. If you plug in $u = 0.73$, the CDF simply tells you that 73% of the total probability lies to the left of 0.73.

Now, let's flip the perspective. What if, instead of asking "how much probability accumulates up to $x$?", we ask the question in reverse: **"At what coordinate $x$ does our accumulated probability reach 73%?"**

For the uniform distribution, the answer is trivially 0.73. But when you apply this exact same reverse lookup to *any* continuous distribution, something remarkable happens.

$$X = F_X^{-1}(U)$$

This strategy is known as **inverse transform sampling**, and its execution is straightforward:

1. Draw a uniform random number $U \sim \mathcal{U}(0,1)$ from your computer's PRNG.
2. Pass that value into the inverse cumulative distribution function $F_X^{-1}(U)$.
3. The resulting output $X$ will be a perfectly valid sample drawn from $p(x)$.

Why does this work so cleanly? The inverse CDF acts as an elastic ruler that stretches and compresses uniform probability levels. Regions of the target distribution containing 10% of the total mass get allocated exactly 10% of the uniform inputs. Steeper parts of the CDF compress uniform inputs into tight clusters, while flat regions spread them out over wider spans.

We took an elementary, uniform decimal, fed it into a deterministic mathematical function, and produced a sample from a complex target distribution. Sampling isn't about creating random mass out of thin air; **sampling is a geometry transformation problem.**

## A Concrete Example: The Exponential Distribution

To see inverse transform sampling in action, let me walk you through building an exponential sampler from scratch. The exponential distribution is widely used to model wait times between independent events, and its PDF is given by $p(x) = \lambda e^{-\lambda x}$ for $x \geq 0$.

Integrating this density gives us a smooth, closed-form CDF: $F(x) = 1 - e^{-\lambda x}$. To find the inverse function, we set $u = 1 - e^{-\lambda x}$ and solve directly for $x$:

$$\begin{aligned} u &= 1 - e^{-\lambda x} \\ e^{-\lambda x} &= 1 - u \\ -\lambda x &= \log(1 - u) \\ x &= -\frac{1}{\lambda} \log(1 - u) \end{aligned}$$

Because $U$ is uniformly distributed on $[0,1]$, the quantity $1 - U$ is also uniformly distributed on $[0,1]$, allowing us to simplify the expression even further. In Python, our exponential generator boils down to a single line:

```python
import numpy as np

def sample_exponential(lambda_param: float, size: int = 1) -> np.ndarray:
    u = np.random.default_rng().random(size)
    return -np.log(u) / lambda_param
```

If you draw a million points with this function and plot their histogram, you will recover an exponential decay curve. The target distribution didn't exist inside Python's random generator; we forged it by warping a uniform line through a logarithm.

## The Journey So Far

We have built a working pipeline from raw deterministic silicon all the way to controlled probability distributions:

$$\boxed{
\text{random bits}
\rightarrow
\text{PRNG}
\rightarrow
U\sim\mathcal{U}(0,1)
\rightarrow
F^{-1}
\rightarrow
X\sim p(x)
}$$

This algebraic trick works brilliantly whenever we can write down $F^{-1}$ explicitly. But what happens when the math gets messy, the inverse CDF has no closed form, or we want to sample complex high-dimensional objects like images?

In **Part 2**, we will explore how rejection sampling saves us when inversion fails, and how this simple idea of transforming randomness directly powers modern generative AI models.
