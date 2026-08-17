---
layout: post
title: "One Uniform Distribution Is Enough, Until the Dimension Grows"
date: 2026-08-18
description: Inverse-CDF, rejection sampling, and importance sampling all turn uniform bits into arbitrary distributions in one dimension. All three die exponentially in high dimension.
categories: GenAI
tags: [ml, generative-modelling, probability, sampling]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---
I ended the last post with a perfect, uncorrelated stream of numbers in $[0,1)$, and promised that every sampler in this series would be built out of it. I should cash in on that promise, albeit for the easy case. I will do this for one dimension, and a target distribution that you already know how to write down. If you're comfortable with CDFs, basic calculus, and the standard Gaussian, you have everything you need.

Say you have $u \sim \mathcal{U}(0,1)$ and you want $x$ from some other distribution entirely, an exponential, a Gaussian, or a bimodal mess you can only evaluate pointwise. The question is how do you get there from a single uniform number?

## Flipping the CDF, politely

Every distribution has a CDF, defined as $F(x) = P(X \le x)$, and every CDF is a function that takes in $x$ and returns a number in $[0,1]$. What happens if you run it backwards? If $U \sim \mathcal{U}(0,1)$, then $X = F^{-1}(U)$ has exactly the distribution you wanted. This is the so-called the probability integral transform!

Lets look at the exponential distribution as the concrete case, since its CDF can be inverted by hand. $F(x) = 1 - e^{-x}$, so $F^{-1}(u) = -\ln(1-u)$. Note that since $1-U$ is itself uniform on $(0,1)$, people usually just write $-\ln(u)$.

```python
import numpy as np

def sample_exponential(rng, n):
    u = rng.uniform(size=n)
    return -np.log(u)
```

{% include figure.liquid path="assets/img/sampling-series/inverse-cdf.svg" class="img-fluid rounded" alt="The exponential CDF F(x) = 1 - e^{-x} plotted with five horizontal uniform draws mapped across to the curve and down to the corresponding x values, illustrating inverse-CDF sampling." caption="Five uniform draws on the vertical axis, mapped through the inverse CDF to exponential draws on the horizontal one. Notice how the x-values crowd near zero, exactly where the exponential density is highest." %}

The figure shows why this works. Where the CDF is steep, a small window of $u$ covers a wide window of $x$, so draws rarely land there. Places where the CDF is flat, the same window of $u$ covers a narrow window of $x$, so draws pile up. The steepness of $F$ is related to the density, and inverting it converts uniform coverage on the $u$-axis directly into the right coverage on the $x$-axis.

Cool! So why doesn't everyone just do this for everything? Because closed-form inverses are rare. Consider a simple counterexample. The Gaussian. Its CDF, $\Phi(x) = \tfrac12\left[1+\mathrm{erf}(x/\sqrt2)\right]$, has no elementary inverse. You can still get $\Phi^{-1}(u)$, but only by numerical approximation, and Gaussians are used often enough in Monte Carlo inner loops that this used to matter a lot.

## A neat trick for Gaussians

Although a single Gaussian's CDF won't invert, a pair of them will. We just need to move away from Cartesian coordinates. Take two independent standard normals $X, Y$. Their joint density is rotationally symmetric, so switch to polar: $R^2 = X^2+Y^2$ and $\theta = \arctan(Y/X)$. It turns out $\theta$ is uniform on $(0, 2\pi)$ and $R^2$ is exponential, and both of those invert by hand, the same trick as the section above. This is the famous [Box-Muller transform](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-29/issue-2/A-Note-on-the-Generation-of-Random-Normal-Deviates/10.1214/aoms/1177706645.full), from Box and Muller's 1958 paper:

$$
R = \sqrt{-2\ln U_1}, \qquad \theta = 2\pi U_2, \qquad X = R\cos\theta, \quad Y = R\sin\theta.
$$

Two uniforms in, two exact standard normals out, no approximation anywhere. The catch is the two transcendental calls, `sin` and `cos`, which used to be the expensive part of this whole operation on older hardware.

[George Marsaglia's fix](https://epubs.siam.org/doi/10.1137/1006063), with Thomas Bray in 1964, removes them by getting $\cos\theta$ and $\sin\theta$ geometrically instead of trigonometrically. Draw a point $(u,v)$ uniformly on the square $[-1,1]^2$, and keep only the ones that land inside the unit circle, where $s = u^2+v^2 \in (0,1)$. For such a point, $u/\sqrt{s}$ and $v/\sqrt{s}$ already are the cosine and sine of its angle, for free, no trig function needed.

```python
def sample_normal_pair(rng):
    while True:
        u, v = rng.uniform(-1, 1, size=2)
        s = u*u + v*v
        if 0 < s < 1:
            scale = np.sqrt(-2*np.log(s)/s)
            return u*scale, v*scale
```

Notice what just happened. To dodge two trig calls, Marsaglia's method throws darts at a square and keeps only the ones that land in the circle. The fraction it keeps is the circle's area over the square's, $\pi/4 \approx 0.785$. That's a rejection sampler (see below)!

## The Ziggurat

So, neither of the two methods above is what your machine actually calls when you run `np.random.normal()`. That's the Ziggurat algorithm, from [Marsaglia and Tsang's 2000 paper](https://www.jstatsoft.org/article/view/v005i08).

The idea is to cover the density with a stack of horizontal rectangles of equal area, layered from a wide short one at the bottom to a tall thin sliver at the top, near the tail. Sampling then amounts to picking a random layer and a random horizontal position in it. Almost all such positions fall under the curve within that layer, and you can check that with one comparison and no function calls at all. Only the rare position that lands in the sliver where a rectangle pokes out past the true density needs a slower fallback. Since that's a small fraction of draws, the amortized cost is close to one comparison and one multiply, faster than either Box–Muller or its polar cousin. This is what current numpy, and most serious scientific libraries, actually run under `standard_normal`.

## Rejection sampling

Lets generalize what Marsaglia's method was already doing, an idea [von Neumann described in general form back in 1951](https://mcnp.lanl.gov/pdf_files/InBook_Computing_1961_Neumann_JohnVonNeumannCollectedWorks_VariousTechniquesUsedinConnectionwithRandomDigits.pdf). Say you have a target density $p(x)$, maybe not even normalized, maybe known only up to a constant that you can't compute. This is often the normal state of affairs in real life. You have a proposal $q(x)$ you know how to sample and evaluate, and a constant $M$ large enough that $p(x) \le M q(x)$ everywhere. Draw $x \sim q$, draw $u \sim \mathcal{U}(0,1)$ independently, and accept $x$ only if $u \le p(x)/(Mq(x))$.

```python
def rejection_sample(rng, p, q_sample, q_pdf, M):
    while True:
        x = q_sample(rng)
        u = rng.uniform()
        if u <= p(x) / (M * q_pdf(x)):
            return x
```

The accepted draws are distributed exactly as $p$, and the overall probability of accepting any given proposal is exactly $1/M$, independent of $x$:

$$
P(\text{accept}) = \int q(x)\,\frac{p(x)}{Mq(x)}\,dx = \frac{1}{M}\int p(x)\,dx = \frac{1}{M}.
$$

{% include figure.liquid path="assets/img/sampling-series/rejection-sampling.svg" class="img-fluid rounded" alt="A bimodal target density p(x) with a wider Gaussian envelope M times q(x) drawn over it, and a scatter of sampled points showing which fall under p(x) and are accepted versus which fall between p(x) and the envelope and are rejected." caption="Rejection sampling against a bimodal target, using a single wide Gaussian as the proposal. Points under the solid curve are kept; points above it but under the dashed envelope are thrown away. Here 1/M is about 41 percent, so well over half of every draw is wasted." %}

The whole method depends on how tight $M$ can be made, which is really a question of how well $q$'s shape matches $p$'s. If you bound a target with a proposal that's much wider or a different shape, as in the figure, $M$ balloons, and $1/M$ collapses. Every draw you throw away costs exactly as much to generate as one you keep.

## Importance sampling

If you push $q$ far enough from $p$, $M$ becomes unknown, or so large that keeping even one out of a million proposals would be luck. Rejection sampling's whole strategy is quite literally _discard what doesn't fit_, which stops being viable in this case.

Importance sampling keeps every draw and fixes the mismatch by reweighting instead! Lets draw $x_1, \dots, x_N$ from $q$, and instead of accepting or rejecting, attach each one a weight $w_i = \tilde p(x_i)/q(x_i)$, where $\tilde p$ is allowed to be an unnormalized version of your target. Then normalize the weights to sum to one and use them directly as an estimator:

$$
\mathbb{E}_p[f(X)] \approx \sum_i \hat w_i\, f(x_i), \qquad \hat w_i = \frac{w_i}{\sum_j w_j}.
$$

```python
def self_normalized_is(x_samples, p_tilde, q_pdf, f):
    w = p_tilde(x_samples) / q_pdf(x_samples)
    w_hat = w / w.sum()
    return np.sum(w_hat * f(x_samples))
```

Look closely at that ratio and notice that any constant multiplying $\tilde p$, in particular whatever normalizing constant $Z$ you didn't want to compute, cancels between the numerator and denominator! This is the first time in this series that the normalizer disappears for free rather than needing to be dodged with cleverness (it won't be the last). Every family of generative model this series eventually covers is, in one way or another, working around the fact that $Z$ is usually the one quantity you cannot get your hands on!

Reweighting instead of discarding sounds like it should be strictly better than rejection, and in one sense it is. You never throw a sample away outright. But a badly matched $q$ doesn't disappear. It just relocates into the weights. If $q$ rarely falls in the region where $p$ actually lives, the few samples that do land there get enormous weights, and everything else gets a weight near zero. Your estimate is then effectively built from one or two data points.

The standard way to see through the disguise is the effective sample size, a diagnostic Leslie Kish worked out for weighted survey estimates back in 1965: $\mathrm{ESS} = (\sum_i w_i)^2 / \sum_i w_i^2$. It ranges from $1$, one dominant weight carrying everything, up to $N$, every weight equal. It answers a concrete question: how many honest draws from $p$ would your $N$ weighted draws from $q$ be worth?

Okay, a brief aside for completeness before the main event. Low-discrepancy sequences ([Halton, 1960](https://eudml.org/doc/131448); [Sobol, 1967](https://www.mathnet.ru/eng/zvmmf7334)) replace i.i.d. uniform draws with deterministic ones engineered to fill space more evenly at every $N$, which tightens both rejection and importance sampling's constants without changing their exponential dependence on dimension. It's a real technique, quasi-Monte Carlo, but it postpones the problem in the next section instead of solving it.

## Cranking up the dimension

Everything above was doing fine in one dimension. Watch what happens if I push it into $d$ dimensions. 

For rejection sampling, lets take the cleanest possible mismatch: target uniform inside the unit ball, proposal uniform on its bounding cube $[-1,1]^d$. The acceptance rate is the volume ratio of ball to cube. For importance sampling, take two Gaussians differing only slightly in scale, target $\mathcal{N}(0, 1.1^2 I_d)$ against proposal $\mathcal{N}(0, I_d)$, and track $\mathrm{ESS}/N$ as $d$ grows.

{% include figure.liquid path="assets/img/sampling-series/curse-of-dimension.svg" class="img-fluid rounded" alt="Two log-scale line plots side by side. Left: rejection sampling acceptance rate for a ball inside its bounding cube, dropping from 1 at dimension 1 to about 10^-10 by dimension 25. Right: effective sample size fraction for importance sampling between two Gaussians differing by 10 percent in standard deviation, dropping from near 1 at dimension 1 to under 2 percent by dimension 200." caption="Both quantities are on a log axis. Rejection sampling's acceptance rate falls below one in a billion by d=25 for a mismatch you'd call negligible in 1D. Importance sampling survives longer, since the two Gaussians here differ by only 10 percent in scale, but the same exponential decay is visible by d=200." %}

The ball-in-cube acceptance rate is already down to 0.25 percent by $d=10$ and to $2.5\times10^{-8}$ by $d=20$. The importance sampling side degrades slower, since a 10 percent scale mismatch is a much gentler test than a ball versus a cube, but by $d=200$ the effective sample size has fallen to under 2 percent of $N$: two hundred thousand weighted draws behaving like four thousand real ones.

This isn't a quirk of these two toy setups by the way. [Snyder, Bengtsson, Bickel, and Anderson (2008)](https://doi.org/10.1175/2008MWR2529.1) worked out the general version of the importance sampling story for particle filters and found that the ensemble size needed to avoid weight collapse scales exponentially with the state dimension, and reported needing on the order of $10^{11}$ particles for a 200-dimensional problem with independent Gaussian errors, a regime not so different from the toy above.

The mechanism behind both plots is the same one. A small per-coordinate mismatch, whether it's a ball versus a cube's corner, or a 10 percent difference in scale, gets multiplied across $d$ independent coordinates. Products of $d$ numbers each a little less than one shrink exponentially in $d$, whether or not any single coordinate looks like a problem. Dimension makes these methods fail on a scale no single coordinate's plot would predict.

I find this the most underrated fact in the foundations of Monte Carlo methods, more so than any individual algorithm in this post. Everything above works by drawing i.i.d. proposals and either keeping the lucky ones or weighting them by how lucky they got. In high dimension, being simultaneously lucky in every one of $d$ directions is not a rare event. It is, for practical purposes, not an event at all.

This post is part of a series on sampling and generative modeling, working up from raw uniform bits to the machinery behind modern generative models. The next post drops the requirement that proposals be independent. A Markov chain that's allowed to wander, correlated step after correlated step, spending its time in proportion to how much probability mass sits wherever it currently is, sidesteps the need to guess a global envelope or a global proposal altogether. That's Metropolis–Hastings, and it's the actual answer to the crisis in the figure above.
