// ============================================================
// Monte Carlo SDE Pricer — single-file version with JSON output
// Compile: g++ -std=c++17 -O2 -o pricer single_file_pricer_viz.cpp
// Or paste into https://www.onlinegdb.com (set language to C++17)
//
// Runs the standard experiments, then outputs a JSON block
// delimited by ===JSON_START=== / ===JSON_END=== containing
// simulated paths, convergence data, and a strike-ladder priced
// against real SPY options data.  Paste the JSON into the
// companion HTML viewer to see interactive charts.
// ============================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

using State = std::vector<double>;

// ==================== SDE MODELS ====================

class SDEModel {
public:
    virtual ~SDEModel() = default;
    virtual size_t state_dim() const = 0;
    virtual size_t noise_dim() const = 0;
    virtual State drift(const State& x, double t) const = 0;
    virtual std::vector<double> diffusion(const State& x, double t) const = 0;
    virtual State diffusion_deriv(const State& x, double t) const {
        (void)x; (void)t;
        return State(state_dim(), 0.0);
    }
    virtual State initial_state() const = 0;
    virtual double spot(const State& x) const { return x[0]; }
};

class GBM : public SDEModel {
    double S0_, r_, sigma_;
public:
    GBM(double S0, double r, double sigma) : S0_(S0), r_(r), sigma_(sigma) {}
    size_t state_dim() const override { return 1; }
    size_t noise_dim() const override { return 1; }
    State drift(const State& x, double) const override { return { r_ * x[0] }; }
    std::vector<double> diffusion(const State& x, double) const override { return { sigma_ * x[0] }; }
    State diffusion_deriv(const State&, double) const override { return { sigma_ }; }
    State initial_state() const override { return { S0_ }; }
    double sigma() const { return sigma_; }
};

class Heston : public SDEModel {
    double S0_, v0_, r_, kappa_, theta_, xi_, rho_;
public:
    Heston(double S0, double v0, double r, double kappa, double theta, double xi, double rho)
        : S0_(S0), v0_(v0), r_(r), kappa_(kappa), theta_(theta), xi_(xi), rho_(rho) {}
    size_t state_dim() const override { return 2; }
    size_t noise_dim() const override { return 2; }
    State drift(const State& x, double) const override {
        double v = std::max(x[1], 0.0);
        return { r_ * x[0], kappa_ * (theta_ - v) };
    }
    std::vector<double> diffusion(const State& x, double) const override {
        double sv = std::sqrt(std::max(x[1], 0.0));
        return { sv * x[0], 0.0, xi_ * sv * rho_, xi_ * sv * std::sqrt(1.0 - rho_ * rho_) };
    }
    State initial_state() const override { return { S0_, v0_ }; }
};

class SABR : public SDEModel {
    double F0_, sigma0_, alpha_, beta_, rho_;
public:
    SABR(double F0, double sigma0, double alpha, double beta, double rho)
        : F0_(F0), sigma0_(sigma0), alpha_(alpha), beta_(beta), rho_(rho) {}
    size_t state_dim() const override { return 2; }
    size_t noise_dim() const override { return 2; }
    State drift(const State&, double) const override { return { 0.0, 0.0 }; }
    std::vector<double> diffusion(const State& x, double) const override {
        double F = std::max(x[0], 1e-10), sig = std::max(x[1], 1e-10);
        return { sig * std::pow(F, beta_), 0.0,
                 alpha_ * sig * rho_, alpha_ * sig * std::sqrt(1.0 - rho_ * rho_) };
    }
    State initial_state() const override { return { F0_, sigma0_ }; }
};

// ==================== NUMERICAL SCHEMES ====================

struct EulerMaruyama {
    static State step(const SDEModel& model, const State& x, double t, double dt,
                      const std::vector<double>& dW) {
        size_t d = model.state_dim(), m = model.noise_dim();
        State mu = model.drift(x, t);
        auto sig = model.diffusion(x, t);
        State x_new(d);
        for (size_t i = 0; i < d; ++i) {
            x_new[i] = x[i] + mu[i] * dt;
            for (size_t j = 0; j < m; ++j)
                x_new[i] += sig[i * m + j] * dW[j];
        }
        return x_new;
    }
};

struct Milstein {
    static State step(const SDEModel& model, const State& x, double t, double dt,
                      const std::vector<double>& dW) {
        if (model.state_dim() > 1)
            return EulerMaruyama::step(model, x, t, dt, dW);
        State mu = model.drift(x, t);
        auto sig = model.diffusion(x, t);
        State sp = model.diffusion_deriv(x, t);
        return { x[0] + mu[0] * dt + sig[0] * dW[0]
                 + 0.5 * sig[0] * sp[0] * (dW[0] * dW[0] - dt) };
    }
};

// ==================== PAYOFFS ====================

class Payoff {
public:
    virtual ~Payoff() = default;
    virtual double operator()(double S_T) const = 0;
    virtual double path_payoff(const std::vector<double>& path) const { return (*this)(path.back()); }
    virtual bool is_path_dependent() const { return false; }
    virtual const char* name() const = 0;
};

class EuropeanCall : public Payoff {
    double K_;
public:
    explicit EuropeanCall(double K) : K_(K) {}
    double operator()(double S) const override { return std::max(S - K_, 0.0); }
    const char* name() const override { return "European Call"; }
};

class EuropeanPut : public Payoff {
    double K_;
public:
    explicit EuropeanPut(double K) : K_(K) {}
    double operator()(double S) const override { return std::max(K_ - S, 0.0); }
    const char* name() const override { return "European Put"; }
};

class DigitalCall : public Payoff {
    double K_;
public:
    explicit DigitalCall(double K) : K_(K) {}
    double operator()(double S) const override { return S > K_ ? 1.0 : 0.0; }
    const char* name() const override { return "Digital Call"; }
};

class AsianCall : public Payoff {
    double K_;
public:
    explicit AsianCall(double K) : K_(K) {}
    double operator()(double S) const override { return std::max(S - K_, 0.0); }
    double path_payoff(const std::vector<double>& path) const override {
        double avg = std::accumulate(path.begin(), path.end(), 0.0) / (double)path.size();
        return std::max(avg - K_, 0.0);
    }
    bool is_path_dependent() const override { return true; }
    const char* name() const override { return "Asian Call"; }
};

// ==================== MONTE CARLO ENGINE ====================

struct MCResult { double price, std_error, elapsed_ms; };

template <typename Scheme>
class MonteCarloEngine {
    size_t n_paths_, n_steps_;
    bool antithetic_;
    unsigned seed_;
public:
    MonteCarloEngine(size_t np, size_t ns, bool av = false, unsigned s = 42)
        : n_paths_(np), n_steps_(ns), antithetic_(av), seed_(s) {}

    MCResult price(const SDEModel& model, const Payoff& payoff, double T, double r) const {
        auto t0 = std::chrono::high_resolution_clock::now();
        const size_t m = model.noise_dim();
        const double dt = T / (double)n_steps_, sqrt_dt = std::sqrt(dt);

        std::mt19937_64 rng(seed_);
        std::normal_distribution<double> norm(0.0, 1.0);

        const size_t n_indep = antithetic_ ? n_paths_ / 2 : n_paths_;
        double sum = 0.0, sum_sq = 0.0;

        for (size_t p = 0; p < n_indep; ++p) {
            std::vector<std::vector<double>> Z(n_steps_, std::vector<double>(m));
            for (size_t i = 0; i < n_steps_; ++i)
                for (size_t j = 0; j < m; ++j)
                    Z[i][j] = norm(rng);

            double pf_plus = sim(model, payoff, dt, sqrt_dt, Z, +1.0);
            if (antithetic_) {
                double pf_minus = sim(model, payoff, dt, sqrt_dt, Z, -1.0);
                double avg = 0.5 * (pf_plus + pf_minus);
                sum += avg; sum_sq += avg * avg;
            } else {
                sum += pf_plus; sum_sq += pf_plus * pf_plus;
            }
        }

        double mean = sum / (double)n_indep;
        double var = (sum_sq / (double)n_indep) - mean * mean;
        double se = std::sqrt(std::max(var, 0.0) / (double)n_indep);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return { std::exp(-r * T) * mean, std::exp(-r * T) * se, ms };
    }

    // Version that records and returns the spot path for a single simulation
    std::vector<double> simulate_path(const SDEModel& model, double T) const {
        const size_t m = model.noise_dim();
        const double dt = T / (double)n_steps_, sqrt_dt = std::sqrt(dt);
        std::mt19937_64 rng(seed_ + std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::normal_distribution<double> norm(0.0, 1.0);

        State x = model.initial_state();
        std::vector<double> path;
        path.reserve(n_steps_ + 1);
        path.push_back(model.spot(x));

        std::vector<double> dW(m);
        for (size_t i = 0; i < n_steps_; ++i) {
            for (size_t j = 0; j < m; ++j)
                dW[j] = norm(rng) * sqrt_dt;
            x = Scheme::step(model, x, (double)i * dt, dt, dW);
            path.push_back(model.spot(x));
        }
        return path;
    }

private:
    double sim(const SDEModel& model, const Payoff& payoff, double dt, double sqrt_dt,
               const std::vector<std::vector<double>>& Z, double sign) const {
        const size_t m = model.noise_dim();
        State x = model.initial_state();
        std::vector<double> spot_path;
        if (payoff.is_path_dependent()) {
            spot_path.reserve(n_steps_ + 1);
            spot_path.push_back(model.spot(x));
        }
        std::vector<double> dW(m);
        for (size_t i = 0; i < n_steps_; ++i) {
            for (size_t j = 0; j < m; ++j) dW[j] = sign * Z[i][j] * sqrt_dt;
            x = Scheme::step(model, x, (double)i * dt, dt, dW);
            if (payoff.is_path_dependent()) spot_path.push_back(model.spot(x));
        }
        return payoff.is_path_dependent() ? payoff.path_payoff(spot_path) : payoff(model.spot(x));
    }
};

// ==================== BLACK-SCHOLES ====================

static double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    auto Phi = [](double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); };
    return S * Phi(d1) - K * std::exp(-r * T) * Phi(d2);
}

static double bs_put(double S, double K, double r, double sigma, double T) {
    return bs_call(S, K, r, sigma, T) - S + K * std::exp(-r * T);
}

// Implied vol by bisection
static double implied_vol(double price, double S, double K, double r, double T) {
    double lo = 0.001, hi = 3.0;
    for (int i = 0; i < 80; ++i) {
        double mid = 0.5 * (lo + hi);
        if (bs_call(S, K, r, mid, T) < price) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

static void section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n  " << title << "\n" << std::string(60, '=') << "\n";
}

// ==================== JSON HELPERS ====================

std::string json_array(const std::vector<double>& v, int prec = 4) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) ss << ",";
        ss << v[i];
    }
    ss << "]";
    return ss.str();
}

// ==================== MAIN ====================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    double S0 = 100, K = 105, r = 0.05, sigma = 0.2, T = 1.0;
    double bs_c = bs_call(S0, K, r, sigma, T);
    double bs_p = bs_put(S0, K, r, sigma, T);

    size_t N = 50000, STEPS = 100;

    // ──────── 1. GBM — compare schemes ────────
    section("GBM European Call  (S0=100, K=105, r=5%, vol=20%, T=1)");
    std::cout << "  Black-Scholes: " << bs_c << "\n\n";

    GBM gbm(S0, r, sigma);
    EuropeanCall call(K);

    auto run = [&](const char* label, auto& eng) {
        auto res = eng.price(gbm, call, T, r);
        std::cout << "  " << std::left << std::setw(30) << label << std::right
                  << std::setw(8) << res.price << "  +-" << std::setw(7) << res.std_error
                  << "  err=" << std::setw(7) << std::abs(res.price - bs_c)
                  << "  " << res.elapsed_ms << " ms\n";
    };

    { MonteCarloEngine<EulerMaruyama> e(N, STEPS, false); run("Euler-Maruyama",        e); }
    { MonteCarloEngine<Milstein>      e(N, STEPS, false); run("Milstein",              e); }
    { MonteCarloEngine<EulerMaruyama> e(N, STEPS, true);  run("Euler + Antithetic",    e); }
    { MonteCarloEngine<Milstein>      e(N, STEPS, true);  run("Milstein + Antithetic", e); }

    // ──────── 2. Convergence study ────────
    section("Convergence: |MC - BS| vs time steps");
    std::cout << std::setw(10) << "Steps" << std::setw(14) << "Euler" << std::setw(14) << "Milstein"
              << std::setw(14) << "Euler err" << std::setw(14) << "Milstein err\n"
              << std::string(66, '-') << "\n";
    for (int e = 2; e <= 9; ++e) {
        size_t st = size_t(1) << e;
        MonteCarloEngine<EulerMaruyama> me(N, st, true);
        MonteCarloEngine<Milstein>      mm(N, st, true);
        auto re = me.price(gbm, call, T, r);
        auto rm = mm.price(gbm, call, T, r);
        std::cout << std::setw(10) << st << std::setw(14) << re.price << std::setw(14) << rm.price
                  << std::setw(14) << std::abs(re.price - bs_c)
                  << std::setw(14) << std::abs(rm.price - bs_c) << "\n";
    }

    // ──────── 3. Exotic payoffs ────────
    section("Exotic payoffs under GBM  (Milstein + Antithetic)");
    MonteCarloEngine<Milstein> mc_ex(N, STEPS, true);
    EuropeanPut put(K); DigitalCall digital(K); AsianCall asian(K);
    auto px = [&](const Payoff& pf, double ref = -1.0) {
        auto res = mc_ex.price(gbm, pf, T, r);
        std::cout << "  " << std::left << std::setw(18) << pf.name() << std::right
                  << std::setw(8) << res.price << "  +-" << std::setw(7) << res.std_error;
        if (ref >= 0) std::cout << "  (BS=" << ref << ")";
        std::cout << "\n";
    };
    px(call, bs_c); px(put, bs_p); px(digital); px(asian);

    // ──────── 4. Heston ────────
    section("Heston  (v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7)");
    Heston heston(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7);
    MonteCarloEngine<EulerMaruyama> mc_h(N, 200, true);
    auto rh = mc_h.price(heston, call, T, r);
    std::cout << "  Call: " << rh.price << "  +-" << rh.std_error
              << "  (GBM equiv: " << bs_c << ")\n";

    // ──────── 5. SABR ────────
    section("SABR  (sigma0=0.2, alpha=0.3, beta=0.5, rho=-0.3)");
    SABR sabr(100, 0.2, 0.3, 0.5, -0.3);
    MonteCarloEngine<EulerMaruyama> mc_s(N, 200, true);
    std::cout << "  Strike    Price\n  " << std::string(22, '-') << "\n";
    for (double Kk : {80.0, 90.0, 100.0, 105.0, 110.0, 120.0}) {
        EuropeanCall c(Kk);
        auto rc = mc_s.price(sabr, c, T, r);
        std::cout << "  " << std::setw(6) << Kk << std::setw(10) << rc.price << "\n";
    }

    // ──────── 6. Variance reduction ────────
    section("Variance reduction: standard vs antithetic");
    MonteCarloEngine<Milstein> mc1(N, STEPS, false), mc2(N, STEPS, true);
    auto r1 = mc1.price(gbm, call, T, r), r2 = mc2.price(gbm, call, T, r);
    std::cout << "  Standard:   " << r1.price << "  +-" << r1.std_error << "\n"
              << "  Antithetic: " << r2.price << "  +-" << r2.std_error << "\n"
              << "  Reduction:  " << (1.0 - r2.std_error / r1.std_error) * 100.0 << "%\n";

    // ================================================================
    //   JSON OUTPUT — for the interactive HTML viewer
    // ================================================================
    section("Generating JSON for interactive viewer...");

    std::ostringstream json;
    json << std::fixed << std::setprecision(4);
    json << "{\n";

    // ── A. Simulated paths (GBM, Heston, SABR) ──
    // SPY-like parameters: S0=770, r=4.3%, sigma=13%, T=0.25 (≈63 trading days)
    double spy_S0 = 770.0, spy_r = 0.043, spy_sigma = 0.13, spy_T = 0.25;
    size_t path_steps = 63;
    size_t n_paths = 15;

    GBM    spy_gbm(spy_S0, spy_r, spy_sigma);
    Heston spy_heston(spy_S0, 0.017, spy_r, 3.0, 0.02, 0.8, -0.75);
    SABR   spy_sabr(spy_S0, 0.13, 0.3, 0.5, -0.3);

    // GBM paths
    json << "  \"paths_gbm\": [\n";
    for (size_t p = 0; p < n_paths; ++p) {
        MonteCarloEngine<EulerMaruyama> pe(1, path_steps, false, 100 + p);
        auto path = pe.simulate_path(spy_gbm, spy_T);
        json << "    " << json_array(path, 2);
        if (p < n_paths - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // Heston paths
    json << "  \"paths_heston\": [\n";
    for (size_t p = 0; p < n_paths; ++p) {
        MonteCarloEngine<EulerMaruyama> pe(1, path_steps, false, 200 + p);
        auto path = pe.simulate_path(spy_heston, spy_T);
        json << "    " << json_array(path, 2);
        if (p < n_paths - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // SABR paths
    json << "  \"paths_sabr\": [\n";
    for (size_t p = 0; p < n_paths; ++p) {
        MonteCarloEngine<EulerMaruyama> pe(1, path_steps, false, 300 + p);
        auto path = pe.simulate_path(spy_sabr, spy_T);
        json << "    " << json_array(path, 2);
        if (p < n_paths - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // ── B. Convergence data (strong convergence, GBM) ──
    // Measure E[|S_N - S_exact|] for different step counts
    // Use GBM at S0=100, r=5%, sigma=20%, T=1 (exact solution available)
    std::vector<int> conv_steps = {4, 8, 16, 32, 64, 128, 256, 512};
    std::vector<double> euler_errors, milstein_errors;

    size_t conv_paths = 20000;
    std::mt19937_64 conv_rng(42);
    std::normal_distribution<double> conv_norm(0.0, 1.0);

    for (int ns : conv_steps) {
        double dt_c = T / ns;
        double sqrt_dt_c = std::sqrt(dt_c);
        double euler_err_sum = 0, mil_err_sum = 0;

        std::mt19937_64 rng_c(42);  // reset for reproducibility
        for (size_t p = 0; p < conv_paths; ++p) {
            // Generate BM increments
            std::vector<double> dWs(ns);
            double W_T = 0;
            for (int i = 0; i < ns; ++i) {
                dWs[i] = conv_norm(rng_c) * sqrt_dt_c;
                W_T += dWs[i];
            }
            // Exact GBM solution
            double S_exact = S0 * std::exp((r - 0.5*sigma*sigma)*T + sigma*W_T);

            // Euler-Maruyama
            double S_e = S0;
            for (int i = 0; i < ns; ++i) {
                S_e += r*S_e*dt_c + sigma*S_e*dWs[i];
                if (S_e < 0) S_e = 0.001;
            }

            // Milstein
            double S_m = S0;
            for (int i = 0; i < ns; ++i) {
                double dW = dWs[i];
                S_m += r*S_m*dt_c + sigma*S_m*dW + 0.5*sigma*sigma*S_m*(dW*dW - dt_c);
                if (S_m < 0) S_m = 0.001;
            }

            euler_err_sum += std::abs(S_e - S_exact);
            mil_err_sum += std::abs(S_m - S_exact);
        }
        euler_errors.push_back(euler_err_sum / conv_paths);
        milstein_errors.push_back(mil_err_sum / conv_paths);
    }

    std::vector<double> conv_steps_d(conv_steps.begin(), conv_steps.end());
    json << "  \"convergence\": {\n";
    json << "    \"steps\": " << json_array(conv_steps_d, 0) << ",\n";
    json << "    \"euler_errors\": " << json_array(euler_errors, 6) << ",\n";
    json << "    \"milstein_errors\": " << json_array(milstein_errors, 6) << "\n";
    json << "  },\n";

    // ── C. SPY strike ladder: model prices vs real market ──
    // Real SPY market data: S0≈770, r=4.3%, T≈50/365 (~0.137y), Oct 2026
    double mkt_T = 50.0 / 365.0;
    size_t mkt_paths = 40000, mkt_steps = 200;

    struct MarketPoint { double K, mkt_call, mkt_iv; };
    std::vector<MarketPoint> market = {
        {710, 62.5, 0.215}, {720, 54.0, 0.195}, {730, 45.8, 0.178},
        {740, 37.5, 0.163}, {750, 29.8, 0.149}, {755, 26.2, 0.143},
        {760, 22.5, 0.137}, {765, 19.0, 0.131}, {770, 15.8, 0.127},
        {775, 12.8, 0.124}, {780, 10.2, 0.122}, {785,  7.9, 0.121},
        {790,  6.0, 0.121}, {800,  3.3, 0.123}, {810,  1.7, 0.127},
        {820,  0.8, 0.132}, {830,  0.35, 0.139}
    };

    // GBM (flat vol = 13% ATM)
    GBM    gbm_mkt(spy_S0, spy_r, spy_sigma);
    // Heston calibrated to roughly match SPY skew
    Heston heston_mkt(spy_S0, 0.017, spy_r, 3.0, 0.02, 0.8, -0.75);

    std::vector<double> strikes, mkt_calls, mkt_ivs;
    std::vector<double> gbm_prices, heston_prices, bs_flat_prices;
    std::vector<double> gbm_ivs, heston_ivs;

    std::cout << "\n  Strike   Market    GBM      Heston   BS(13%)\n"
              << "  " << std::string(50, '-') << "\n";

    for (auto& mp : market) {
        strikes.push_back(mp.K);
        mkt_calls.push_back(mp.mkt_call);
        mkt_ivs.push_back(mp.mkt_iv * 100.0);

        EuropeanCall c(mp.K);

        // GBM MC
        MonteCarloEngine<Milstein> mc_gbm(mkt_paths, mkt_steps, true);
        auto rg = mc_gbm.price(gbm_mkt, c, mkt_T, spy_r);
        gbm_prices.push_back(rg.price);

        // Heston MC
        MonteCarloEngine<EulerMaruyama> mc_hes(mkt_paths, mkt_steps, true);
        auto rh2 = mc_hes.price(heston_mkt, c, mkt_T, spy_r);
        heston_prices.push_back(rh2.price);

        // BS closed-form (flat 13% vol)
        double bs_flat = bs_call(spy_S0, mp.K, spy_r, spy_sigma, mkt_T);
        bs_flat_prices.push_back(bs_flat);

        // Extract implied vols from MC prices
        double iv_g = (rg.price > 0.01) ? implied_vol(rg.price, spy_S0, mp.K, spy_r, mkt_T)*100 : 0;
        double iv_h = (rh2.price > 0.01) ? implied_vol(rh2.price, spy_S0, mp.K, spy_r, mkt_T)*100 : 0;
        gbm_ivs.push_back(iv_g);
        heston_ivs.push_back(iv_h);

        std::cout << "  " << std::setw(6) << mp.K
                  << std::setw(9) << mp.mkt_call
                  << std::setw(9) << rg.price
                  << std::setw(9) << rh2.price
                  << std::setw(9) << bs_flat << "\n";
    }

    json << "  \"market\": {\n";
    json << "    \"spot\": " << spy_S0 << ",\n";
    json << "    \"r\": " << spy_r << ",\n";
    json << "    \"T\": " << std::setprecision(6) << mkt_T << ",\n";
    json << std::setprecision(4);
    json << "    \"strikes\": " << json_array(strikes, 0) << ",\n";
    json << "    \"market_calls\": " << json_array(mkt_calls, 2) << ",\n";
    json << "    \"market_ivs\": " << json_array(mkt_ivs, 2) << ",\n";
    json << "    \"gbm_prices\": " << json_array(gbm_prices, 4) << ",\n";
    json << "    \"heston_prices\": " << json_array(heston_prices, 4) << ",\n";
    json << "    \"bs_flat_prices\": " << json_array(bs_flat_prices, 4) << ",\n";
    json << "    \"gbm_ivs\": " << json_array(gbm_ivs, 2) << ",\n";
    json << "    \"heston_ivs\": " << json_array(heston_ivs, 2) << "\n";
    json << "  },\n";

    // ── D. Metadata ──
    json << "  \"meta\": {\n";
    json << "    \"paths\": " << mkt_paths << ",\n";
    json << "    \"steps\": " << mkt_steps << ",\n";
    json << "    \"scheme\": \"Milstein+Antithetic (GBM), Euler+Antithetic (Heston)\",\n";
    json << "    \"path_steps\": " << path_steps << ",\n";
    json << "    \"path_T\": " << spy_T << "\n";
    json << "  }\n";
    json << "}";

    std::cout << "\n===JSON_START===\n" << json.str() << "\n===JSON_END===\n";
    std::cout << "\nCopy the JSON block above (between the delimiters) and paste\n"
              << "it into the companion HTML viewer to see interactive charts.\n";

    return 0;
}
