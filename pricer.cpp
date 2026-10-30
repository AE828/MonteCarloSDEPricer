// ============================================================
// Monte Carlo SDE Pricer — single-file version
// Compile: g++ -std=c++17 -O2 -o pricer single_file_pricer.cpp
// Or paste into https://www.onlinegdb.com (set language to C++17)
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

// dS = r*S*dt + sigma*S*dW
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
    double r() const { return r_; }
    double sigma() const { return sigma_; }
};

// dS = r*S*dt + sqrt(v)*S*dW1,  dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW2,  corr=rho
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
    double r() const { return r_; }
};

// dF = sigma*F^beta*dW1,  dsigma = alpha*sigma*dW2,  corr=rho
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

// ==================== BLACK-SCHOLES (ground truth) ====================

static double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    auto Phi = [](double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); };
    return S * Phi(d1) - K * std::exp(-r * T) * Phi(d2);
}

static double bs_put(double S, double K, double r, double sigma, double T) {
    return bs_call(S, K, r, sigma, T) - S + K * std::exp(-r * T);
}

static void section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n  " << title << "\n" << std::string(60, '=') << "\n";
}

// ==================== MAIN ====================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    double S0 = 100, K = 105, r = 0.05, sigma = 0.2, T = 1.0;
    double bs_c = bs_call(S0, K, r, sigma, T);
    double bs_p = bs_put(S0, K, r, sigma, T);

    size_t N = 50000, STEPS = 100;

    // 1. GBM — compare schemes
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

    // 2. Convergence study
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

    // 3. Exotic payoffs
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

    // 4. Heston
    section("Heston  (v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7)");
    Heston heston(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7);
    MonteCarloEngine<EulerMaruyama> mc_h(N, 200, true);
    auto rh = mc_h.price(heston, call, T, r);
    std::cout << "  Call: " << rh.price << "  +-" << rh.std_error
              << "  (GBM equiv: " << bs_c << ")\n";

    // 5. SABR
    section("SABR  (sigma0=0.2, alpha=0.3, beta=0.5, rho=-0.3)");
    SABR sabr(100, 0.2, 0.3, 0.5, -0.3);
    MonteCarloEngine<EulerMaruyama> mc_s(N, 200, true);
    std::cout << "  Strike    Price\n  " << std::string(22, '-') << "\n";
    for (double Kk : {80.0, 90.0, 100.0, 105.0, 110.0, 120.0}) {
        EuropeanCall c(Kk);
        auto rc = mc_s.price(sabr, c, T, r);
        std::cout << "  " << std::setw(6) << Kk << std::setw(10) << rc.price << "\n";
    }

    // 6. Variance reduction
    section("Variance reduction: standard vs antithetic");
    MonteCarloEngine<Milstein> mc1(N, STEPS, false), mc2(N, STEPS, true);
    auto r1 = mc1.price(gbm, call, T, r), r2 = mc2.price(gbm, call, T, r);
    std::cout << "  Standard:   " << r1.price << "  +-" << r1.std_error << "\n"
              << "  Antithetic: " << r2.price << "  +-" << r2.std_error << "\n"
              << "  Reduction:  " << (1.0 - r2.std_error / r1.std_error) * 100.0 << "%\n";

    return 0;
}
