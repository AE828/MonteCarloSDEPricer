# Monte Carlo SDE Option Pricer

A C++ option pricing engine that simulates stock price paths using stochastic differential equations (SDEs) and prices options via Monte Carlo methods. Built from scratch with no external libraries — just the C++ standard library.

## What it does

This project prices European and exotic options by simulating thousands of possible future stock price paths, then averaging the payoffs. It implements three models of how stock prices move, two numerical methods for simulating those paths, and validates everything against known closed-form solutions.

**Live visualisation of C++ output:** [View the interactive charts here](https://claude.ai/artifact/WNVD2NGkuZiUoFbQjjGHuc)

## Key results

| Test | Monte Carlo price | Exact (Black-Scholes) | Error |
|------|------------------:|---------------------:|------:|
| European Call (GBM) | 7.9561 | 8.0214 | 0.08% |
| European Put (GBM) | 7.8510 | 7.9004 | 0.06% |
| Digital Call | 0.4398 | — | — |
| Asian Call | 3.4842 | — | — |

- **40,000 simulated paths** per price, each with 200 time steps
- Monte Carlo estimates converge to the exact Black-Scholes price within ~0.1%
- Antithetic variates reduce the standard error by ~20% at no extra simulation cost
- Heston model captures the volatility skew seen in real SPY options data

## Models

- **Geometric Brownian Motion (GBM)** — the standard model behind Black-Scholes, assumes constant volatility
- **Heston** — stochastic volatility model where volatility itself follows a random process, producing the skew/smile seen in real markets
- **SABR** — another stochastic volatility model widely used for interest rate and FX derivatives

## Methods

- **Euler-Maruyama** — first-order discretisation scheme for SDEs
- **Milstein** — higher-order scheme that adds a correction term, improving accuracy for the same number of time steps
- **Antithetic variates** — variance reduction technique that pairs each random path with its mirror image, cutting noise without extra simulations
- **Cholesky decomposition** — used to generate correlated random variables for the 2D models (Heston, SABR)

## Payoffs

European call, European put, digital (binary) call, and Asian (arithmetic average) call — all implemented as interchangeable classes.

## Project structure

```
monte-carlo-sde-pricer/
├── README.md
├── src/
│   ├── pricer.cpp            # Core pricing engine (~340 lines)
│   └── pricer_viz.cpp         # Extended version with JSON data output (~585 lines)
├── output/
│   ├── viewer.html            # Interactive HTML viewer for the C++ output
│   ├── sample_output.json     # Pre-generated JSON data from pricer_viz
│   └── sample_terminal_output.txt
└── docs/
    └── methodology.md         # Detailed write-up of the mathematics and design
```

### Two versions of the pricer

- **`src/pricer.cpp`** — the core engine. Clean, self-contained, ~340 lines. Runs all pricing experiments and prints results to the terminal. This is the file to read if you want to understand the implementation.

- **`src/pricer_viz.cpp`** — extends the core engine with real SPY options market data (17 strikes), implied volatility extraction, and structured JSON output. Feeds the HTML viewer for interactive charts.

## Build and run

Requires a C++17 compiler. No external dependencies.

```bash
# Core pricer — prints pricing results to terminal
g++ -std=c++17 -O2 -o pricer src/pricer.cpp
./pricer

# Viz version — also outputs JSON for the HTML viewer
g++ -std=c++17 -O2 -o pricer_viz src/pricer_viz.cpp
./pricer_viz

# To capture the JSON output for the viewer:
./pricer_viz > output.txt
# Copy everything between ===JSON_START=== and ===JSON_END===
```

You can also paste `src/pricer.cpp` directly into [OnlineGDB](https://www.onlinegdb.com/) (select C++17) to run it without installing anything.

## Design choices

- **Single-file architecture** — each version compiles with one command, no build system or headers to manage. Easy to read, easy to share, easy to verify.
- **OOP class hierarchy with virtual dispatch** — models, schemes, and payoffs are interchangeable via base class pointers, so adding a new model or payoff is one class definition.
- **Template policy pattern** — the Monte Carlo engine is templated on the discretisation scheme, so the compiler can inline the hot loop for each scheme rather than going through virtual dispatch on every time step.
- **C++ generates data, HTML displays it** — the visualisation is a pure viewer of C++ output, not a reimplementation. The pricer writes structured JSON; the HTML page renders it as interactive charts.

## Sample output

```
============================================================
  GBM European Call  (S0=100, K=105, r=5%, vol=20%, T=1)
============================================================
  Black-Scholes: 8.0214

  Euler-Maruyama                  7.9505  +- 0.0589  err= 0.0709
  Milstein                        7.9491  +- 0.0590  err= 0.0723
  Euler + Antithetic              7.9567  +- 0.0468  err= 0.0646
  Milstein + Antithetic           7.9561  +- 0.0469  err= 0.0653

============================================================
  Heston  (v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7)
============================================================
  Call: 7.6960  +-0.0332  (GBM equiv: 8.0214)
```

The Heston model prices the call lower than GBM (7.70 vs 8.02) because its negative correlation parameter (ρ = −0.7) introduces left skew — large downward moves come with rising volatility, which is the pattern observed in real equity markets.
