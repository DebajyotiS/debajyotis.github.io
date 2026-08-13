---
layout: post
title: "How Do We Sample Anything at All? (Part 2: Rejection, Flows, and GenAI)"
date: 2026-08-13
description: Moving from simple inverse transform sampling to rejection sampling, multi-dimensional distributions, and modern generative AI models.
categories: GenAI
tags: [ml, generative-modelling, deep-learning]
featured: false
giscus_comments: false
related_posts: true
---

In [Part 1]({% post_url 2026-08-13-how_we_sample %}), we answered a fundamental question: how do we turn deterministic computer bits into samples from a probability distribution?

We saw that by running the cumulative distribution function (CDF) in reverse, we can take a simple uniform float $U \sim \mathcal{U}(0,1)$ and transform it into a sample from a target distribution via $X = F^{-1}(U)$.

That process feels so seamless that you might wonder why sampling remains an active area of research. The catch is that we quietly hid the hardest step inside $F^{-1}$.

## Where Simple Inversion Runs into a Wall

For basic distributions like the exponential, isolating $x$ takes a few lines of basic high school algebra. But for many crucial distributions, the inverse CDF has no closed-form expression. The standard Gaussian distribution is a classic example:

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
$$

While we can compute its integral numerically, there is no neat algebraic function for $F^{-1}(u)$. Calling a complex numerical approximation inside a loop millions of times gets slow very quickly.

When direct inversion breaks down, we have to get creative. Instead of attempting to invert the target distribution directly, we construct indirect processes that output the target shape as a byproduct.

## Rejection Sampling: Filtering for the Right Shape

Imagine that I want samples from some awkward distribution $p(x)$, but I know how to sample from an easier distribution $q(x)$. Suppose I can draw points from $q$, and I know that $p(x) \leq Mq(x)$ for some constant $M$.

Now I can play a slightly strange game:

1. Sample a candidate $x$ from the easy distribution $q(x)$.
2. Generate another uniform random number $U \sim \mathcal{U}(0,1)$.
3. Keep the candidate only if $U < \frac{p(x)}{Mq(x)}$. Otherwise, throw it away and try again.

This sounds terribly inefficient, and in high dimensions, it often is. But look at what is happening geometrically: we are generating points under an easy curve and selectively carving away excess height until the remaining points match the shape of $p(x)$.

```
First, sample candidate x ~ q(x)
│
▼
Draw uniform check U ~ Uniform(0,1)
│
▼
Is U < p(x) / (M * q(x))?
├── No  ──► Reject and try again
└── Yes ──► Accept sample!

```

Whether through direct inversion or geometric filtering, our strategy remains unchanged. We don't always need to know how to directly generate the distribution we want; sometimes it is enough to generate something easy and then **filter or transform it until the survivors have the right distribution.**

## We Have Been Transforming Randomness This Whole Time

At this point, let's zoom out and review the underlying pattern:

$$
\boxed{
\text{simple randomness}
\rightarrow
\text{transformation}
\rightarrow
\text{desired randomness}
}
$$

Our computer starts with a deterministic algorithm and an initial seed, producing pseudo-random bits. We scale those bits into a simple canonical distribution $U \sim \mathcal{U}(0,1)$, and then apply a deterministic mathematical transformation.

This single realization unlocks the door to high-dimensional machine learning. Because once we move from one-dimensional statistics to complex multi-dimensional spaces, this exact observation becomes the engine driving modern generative modeling.

## From Random Numbers to Generative Models

So far we have been talking about a single scalar random variable. But nothing says our random input has to be one-dimensional. We can instead start with a vector of simple noise:

$$
\mathbf{z} \sim p(\mathbf{z}) \quad \text{where, for instance,} \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Now imagine that we replace our basic single-variable equation with a massive, parameterized neural network:

$$
\mathbf{x} = f_\theta(\mathbf{z})
$$

If $f_\theta$ is designed and trained correctly, a simple Gaussian blob in latent space can be stretched, folded, and warped into a highly intricate distribution over images, text, or audio waveforms.

This is the central concept behind **normalising flows**. Instead of learning how to directly sample from a complicated multi-modal image distribution, we learn an invertible neural network that maps a simple Gaussian distribution straight into it:

$$
\mathbf{z} \sim p_Z(\mathbf{z}) \quad \longrightarrow \quad \mathbf{x} = f(\mathbf{z}) \sim p_X(\mathbf{x})
$$

This is inverse transform sampling grown up. In one dimension, our transformation was $x = F^{-1}(u)$. In a normalising flow, we learn a high-dimensional equivalent of that same mapping function.

And the story doesn't stop there. Other generative frameworks use different mechanics to solve the exact same core problem:

* **Generative Adversarial Networks (GANs)** train a generator network to transform a simple random noise vector into realistic outputs, using a discriminator network to guide the shape of that transformation.
* **Diffusion Models** take a step-by-step route, gradually transforming pure Gaussian noise into structured data over hundreds of small denoising steps.

The algorithms look very different on the surface, but the fundamental conceptual question remains identical across all of them:

> **How can I transform simple randomness into the kind of complex randomness I actually want?**

## The Whole Story in One Picture

We can now map out the complete journey across both posts:

$$
\boxed{
\text{random bits}
\rightarrow
\text{PRNG}
\rightarrow
U\sim\mathcal{U}(0,1)
\rightarrow
\text{Transformation } f(\mathbf{z})
\rightarrow
\mathbf{X}\sim p(\mathbf{x})
}
$$

The first step produces something that behaves randomly. The second step gives us a convenient canonical distribution like the uniform or standard normal. The final step—whether handled by an inverse CDF, a rejection filter, or a 100-million-parameter neural network—turns that plain noise into structured samples.

That is the essence of sampling. Once you see it as a geometry transformation problem, the vast landscape of modern generative AI starts to look like elegant variations on the exact same game.
