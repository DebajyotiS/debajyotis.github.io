---
layout: post
title: "Your Random Number Generator Is Not Random"
date: 2026-08-14
description: Every sample in this series is a deterministic function of uniform bits, where those bits come from and why that's fine.
categories: GenAI
tags: [ml, generative-modelling, probability, rng]
featured: false
giscus_comments: false
related_posts: true
toc:
  beginning: true
  sidebar: true
---
Say you fire up your favourite programming language, and a suitable random number generator (RNG) package therein.
Call `random.seed(42)` (or something equivalent to set the random seed), draw a number, restart the program, call `random.seed(42)` again, and you get the exact same number back. Do it a thousand times and you get the exact same thousand numbers, in the exact same order, forever. If you had to design a definition of "not random" from scratch, perfect on-demand reproducibility would be a good place to start.

And yet this is the thing every probabilistic method in machine learning is built on. This post is for readers comfortable with Python and basic probability who have called a `seed()` function more times than they've asked what it does underneath. What it does, it turns out, is short, mechanical, because every other post in this series (every sampler, every Markov chain, every diffusion model's noise, every flow's base distribution) is going to be a function of what comes out of it.

## A generator is actually just a short program

If you strip away the language bindings, a random number generator is simply a recurrence relation. The oldest and simplest useful one is the so called linear congruential generator:

$$
X_{n+1} = (a X_n + c) \bmod m
$$

Say you pick a multiplier $a$, an increment $c$, a modulus $m$, and a starting value $X_0$ called the seed. This fixes the whole future sequence. To turn an integer $X_n \in \{0, \dots, m-1\}$ into something that looks like a draw from $\mathcal{U}(0,1)$, you simply divide: $u_n = X_n / m$.

Here's a small enough example to run by hand. Take $a=5$, $c=3$, $m=16$, seed $X_0=1$:

```python
def lcg(seed, a, c, m, n):
    x = seed
    out = []
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x)
    return out

lcg(seed=1, a=5, c=3, m=16, n=20)
# [8, 11, 10, 5, 12, 15, 14, 9, 0, 3, 2, 13, 4, 7, 6, 1, 8, 11, 10, 5]
```

Notice that it generates sixteen(!) numbers, and then it repeats from the start. It visited every integer from 0 to 15 exactly once before cycling. This is not an accident. It's a consequence of the particular $a$ and $c$ chosen (the Hull–Dobell theorem gives the exact conditions, if you were keen on following up), and it's the best you could possibly ask of a generator with sixteen possible states. This is the entire mechanism. `numpy.random`, your language's standard library, the generator inside PyTorch: all of it is this idea, with a modulus around $2^{64}$ or larger and considerably more care put into choosing $a$ and $c$.

## A generator can be bad

So, a generator with modulus $m$ can produce at most $m$ distinct values before it repeats. Then the first thing you want is a **long period**: you should exhaust your Monte Carlo budget long before the sequence loops back on itself. That part is easy to check by construction.

The harder property is **equidistribution**, i.e. single outputs $u_n$ should be spread uniformly over $[0,1)$, and so should *tuples* of consecutive outputs $(u_n, u_{n+1}, \dots, u_{n+d-1})$ over $[0,1)^d$, for every $d$ you might care about. A generator can pass every check you throw at single numbers and still be badly broken two or three draws at a time, because the recurrence that generates $u_{n+1}$ from $u_n$ leaves a fingerprint that only shows up when you look at them together.

## Case Study: RANDU

In the 1960s, IBM shipped a generator called RANDU with its scientific subroutine library. It's a pure multiplicative generator (no increment) with

$$
X_{n+1} = a X_n \bmod m, \qquad a = 65539, \quad m = 2^{31}.
$$

```python
def randu(seed, n):
    a, m = 65539, 2**31
    x = seed
    out = []
    for _ in range(n):
        x = (a * x) % m
        out.append(x / m)
    return out
```

RANDU was the default generator on mainframes for over a decade, cited in a huge number of published Monte Carlo results. It looks completely fine if you plot pairs of consecutive draws $(u_n, u_{n+1})$: a uniform speckle over the unit square, no visible structure. The left panel below is four thousand such pairs.

{% include figure.liquid path="assets/img/sampling-series/randu-planes.svg" class="img-fluid rounded" alt="Left panel: a scatter plot of consecutive RANDU pairs (u_n, u_n+1) filling the unit square uniformly with no visible pattern. Right panel: the same draws plotted as u_n against the combination 9u_n - 6u_n+1 + u_n+2, collapsing onto exactly fifteen horizontal lines at integer heights from -5 to 9." caption="Four thousand consecutive draws from RANDU. Pairs look uniform (left). The same points, viewed through the one combination RANDU can't hide, collapse onto exactly fifteen lines (right)." %}

The right panel is the same four thousand draws, plotted differently: $u_n$ on the horizontal axis, and $9u_n - 6u_{n+1} + u_{n+2}$ on the vertical one. Every single point lands on one of fifteen horizontal lines, at integer heights. Not approximate! Exactly, to floating-point precision!

If you are curious, the reason is three lines of algebra. Notice that $a - 3 = 65536 = 2^{16}$, so

$$
(a-3)^2 = 2^{32} = 2 \cdot 2^{31} \equiv 0 \pmod{2^{31}},
$$

which expands to $a^2 - 6a + 9 \equiv 0 \pmod{m}$. Since RANDU has no increment, $X_{n+1} = aX_n \bmod m$ and $X_{n+2} = a^2 X_n \bmod m$, so for every $n$:

$$
X_{n+2} - 6X_{n+1} + 9X_n \equiv (a^2 - 6a + 9)\,X_n \equiv 0 \pmod{m}.
$$

That's an _exact_ linear relation among any three consecutive outputs, baked into the choice of $a$. It forces every triple $(u_n, u_{n+1}, u_{n+2})$ onto one of a handful of parallel planes in the unit cube, a count you can work out from how far $9u_n - 6u_{n+1} + u_{n+2}$ can range, which is exactly fifteen. George Marsaglia published this in 1968, a decade after RANDU had shipped inside statistical packages used across the sciences.

I think the RANDU story usually gets told wrong, if at all, as a cautionary tale about one bad multiplier. The actual failure is that an entire scientific community spent a decade treating "passes the tests we happened to run" as equivalent to "is fine," and those aren't the same claim unless your tests are exhaustive, which they never are. You could object that this is unfair. Bounding a generator's behavior in every dimension it might ever get used in isn't computationally possible, so some amount of `trust-until-broken` is unavoidable. That's true, and it doesn't change what you should do with it, which is treat "nothing's broken yet" as a status report, not a guarantee.

## Passing a test battery is not proof of anything

The modern version of "checking anyone thought to run" is [TestU01](https://simul.iro.umontreal.ca/testu01/tu01.html), Pierre L'Ecuyer's suite of statistical tests, bundled into batteries of increasing severity: SmallCrush, Crush, and BigCrush. The last one runs over a hundred distinct tests, among them birthday spacings, linear complexity, and matrix rank, each looking for a different flavor of structure a generator's output shouldn't have.

"Passing BigCrush" means something weaker than it sounds. There are no test in that specific collection that has detected a deviation from uniform, at whatever significance threshold you chose, on the sequences you happened to feed it. You should know that this is not a proof that the generator is free of exploitable structure, only that the structure isn't one of the roughly hundred kinds anyone has written a test for yet. RANDU would have failed almost every test in BigCrush catastrophically. But the tests that existed in the 1960s were mostly one- and two-dimensional, which were exactly the dimensions in which RANDU looks fine.

## What do people use now?

Currently, to my knowledge, three generators dominate current practice. They trade off state size, speed, and how much scrutiny they've survived.

The **Mersenne Twister** (MT19937) has an enormous period, $2^{19937}-1$, and was the default generator in Python, R, and NumPy for years. It has two soft spots. Its 624 words of internal state can be reconstructed exactly from 624 consecutive outputs, so it has no place near anything adversarial, and it fails a handful of BigCrush's tests, mostly ones sensitive to linear structure, a smaller and subtler echo of RANDU's own problem.

**PCG** (permuted congruential generator) runs an ordinary LCG internally, then applies a permutation to the output before you see it, which erases the linear structure that gives LCGs away. It passes BigCrush, needs far less state than MT19937, and is what `numpy.random.default_rng()` gives you today!

The **xoshiro/xoroshiro** family (Blackman and Vigna) generates by shifting, rotating, and XOR-ing a small register, which makes it fast with only 128–256 bits of state, and it also clears BigCrush.

Here's the practical upshot - If you take one thing from this section: prefer `rng = np.random.default_rng(seed)` and pass `rng` around explicitly, over the older global `np.random.seed(seed)`. A single shared global generator means any library you import that draws numbers behind your back (and several will) perturbs a stream you thought was yours. And if you ever need several independent streams at once, parallel chains, say, both PCG and xoshiro support jumping ahead to a distant point in the sequence rather than juggling multiple arbitrary seeds.

## Why not just use real randomness?

Believe it or not, your computer has access to unpredictable entropy: thermal noise sampled by `/dev/random`, the CPU's `RDRAND` instruction, purpose-built hardware like quantum random number generators. It's tempting to ask why we bother with elaborate deterministic recurrences when a real physical process is sitting right there.

It's because switching to it costs you reproducibility.  You can't replay a specific run or a specific bad sample for debugging, because the source doesn't remember what it produced, and most hardware entropy pools are too slow or rate-limited to feed a Monte Carlo loop anyway. But the mismatch runs deeper than convenience. What a Monte Carlo method needs is equidistribution across however many dimensions it uses at once (precisely the property TestU01 measures and PCG and xoshiro are engineered against). Hardware entropy sources are engineered against a different property, unpredictability to an adversary, which happens to get called "more random" in casual conversation without making it any better suited to this particular job than a generator that's actually been checked against a hundred specific statistical tests.

## Where this leaves us

Modern generators pass every test anyone has thought to run against them. So did RANDU, once. That's not a reason for alarm: it's an epistemic position a deterministic recurrence puts you in.

The next post starts from a perfect, uncorrelated stream of these numbers and asks how to turn it into a sample from an arbitrary distribution. It turns out that even with a flawless generator, the obvious methods fail as you try to scale up the dimensions.
