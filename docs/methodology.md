[methodology.md](https://github.com/user-attachments/files/34194531/methodology.md)
# Methodology

## Overview

This project prices options by simulating possible future stock price paths and averaging the resulting payoffs. The core idea: if you simulate enough paths, the average payoff converges to the fair price of the option (by the law of large numbers).

The mathematical framework combines three ingredients:

1. **A model** describing how the stock price evolves over time (the SDE)
2. **A numerical scheme** for stepping the model forward in discrete time steps
3. **A payoff function** that determines the option's value at expiry

## Stock price models

### Geometric Brownian Motion (GBM)

The simplest model — the stock price grows at a drift rate with constant volatility:

```
dS = r·S·dt + σ·S·dW
```

where `r` is the risk-free rate, `σ` is constant volatility, and `dW` is a random Brownian increment. This is the model underlying the Black-Scholes formula, which provides an exact closed-form price. We use Black-Scholes as ground truth to validate the Monte Carlo engine.

### Heston stochastic volatility

In reality, volatility is not constant — it rises during sell-offs and falls during calm markets. The Heston model captures this by making volatility itself a random process:

```
dS = r·S·dt + √v·S·dW₁
dv = κ(θ - v)·dt + ξ·√v·dW₂
```

where `v` is the instantaneous variance, `κ` is the speed at which variance reverts to its long-run mean `θ`, `ξ` controls the volatility of volatility, and `dW₁` and `dW₂` are correlated Brownian motions with correlation `ρ`.

The negative correlation (`ρ < 0`) produces the "volatility skew" observed in equity options: when the stock drops, volatility rises, making deep out-of-the-money puts more expensive than a constant-volatility model would predict.

We calibrate the Heston parameters to approximately match real SPY option prices observed in October 2026: `v₀ = 0.017, κ = 3.0, θ = 0.02, ξ = 0.8, ρ = −0.75`.

### SABR

The SABR model is used for interest rate and FX options. It models forward prices with stochastic volatility:

```
dF = σ·F^β·dW₁
dσ = α·σ·dW₂
```

where `β` controls the backbone (how the local volatility depends on the price level) and `α` is the vol-of-vol. Like Heston, the two Brownian motions are correlated.

## Numerical schemes

SDEs generally don't have closed-form solutions (GBM is a rare exception). We approximate the continuous-time dynamics by stepping forward in small discrete time increments.

### Euler-Maruyama

The simplest first-order scheme — a direct discretisation of the SDE:

```
S(t+Δt) = S(t) + a(S,t)·Δt + b(S,t)·√Δt·Z
```

where `a` and `b` are the drift and diffusion functions of the SDE, and `Z` is a standard normal random variable. This has **strong convergence order 0.5**, meaning the simulation error scales as `O(√Δt)`.

### Milstein

Adds a correction term involving the derivative of the diffusion coefficient:

```
S(t+Δt) = S(t) + a·Δt + b·√Δt·Z + ½·b·b'·Δt·(Z² - 1)
```

This achieves **strong convergence order 1.0** for one-dimensional SDEs — the error scales as `O(Δt)`, so doubling the number of steps roughly halves the error (vs the square root improvement for Euler-Maruyama). The convergence table in the output demonstrates this empirically.

For multi-dimensional SDEs (Heston, SABR), the standard Milstein scheme requires iterated stochastic integrals (Lévy areas) which are hard to simulate. We use Euler-Maruyama for these models.

## Variance reduction

### Antithetic variates

For each random path driven by normal draws `Z₁, Z₂, ...`, we also simulate the "mirror" path using `-Z₁, -Z₂, ...` and average the two payoffs. Since these paths are negatively correlated, the variance of the average is lower than simulating two independent paths.

In our tests, antithetic variates reduce the confidence interval width by ~20% with no additional random number generation — effectively getting more accuracy for free.

## Implied volatility

To compare models against real market data, we extract the implied volatility from each model's price. The implied volatility is the constant volatility you would need to plug into the Black-Scholes formula to reproduce that price.

We solve for it numerically using bisection: search for the `σ` where `BS(σ) = observed price`, starting from a bracket of `[0.001, 5.0]` and iterating until convergence.

The implied volatility "smile" (or "skew" for equities) is the plot of implied vol against strike price. A flat smile means the model agrees with Black-Scholes at all strikes; a skewed smile means the model assigns different volatilities to different strikes, which is what real markets show.

## Validation

- **GBM vs Black-Scholes**: Monte Carlo prices converge to the exact Black-Scholes price within ~0.1% using 40,000 paths × 200 steps
- **Convergence analysis**: Error decreases as the number of time steps increases, with Milstein converging faster than Euler-Maruyama as expected from theory
- **SPY market comparison**: The Heston model reproduces the shape of the implied volatility skew from real SPY options, while GBM produces a flat line (as expected — constant vol means constant implied vol)

## C++ design

The implementation uses a class hierarchy with virtual functions for models and payoffs:

```
SDEModel (base) ─── GBM
                ├── Heston
                └── SABR

Payoff (base) ───── EuropeanCall
              ├──── EuropeanPut
              ├──── DigitalCall
              └──── AsianCall
```

The Monte Carlo engine is a template class parameterised on the discretisation scheme:

```cpp
template <typename Scheme>
class MonteCarloEngine { ... };
```

This means the compiler generates specialised code for each scheme (Euler, Milstein), avoiding the overhead of virtual dispatch in the innermost simulation loop while keeping the model/payoff dispatch flexible.

Correlated Brownian motions for Heston and SABR are generated via Cholesky decomposition of the correlation matrix.
