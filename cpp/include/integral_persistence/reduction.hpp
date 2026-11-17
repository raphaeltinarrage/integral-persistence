#pragma once

#include "api.hpp"
#include "lazy_engine.hpp"
#include <flint/fmpz.h>
#include <flint/fmpz_factor.h>
#include <set>

namespace integral_native {

/** FLINT objects have deterministic lifetimes even when certification fails. */
struct FlintInteger {
    fmpz_t value;
    explicit FlintInteger(const mpz_class& integer) {
        fmpz_init(value); fmpz_set_mpz(value, integer.get_mpz_t());
    }
    ~FlintInteger() { fmpz_clear(value); }
    FlintInteger(const FlintInteger&) = delete;
};
struct FlintFactors {
    fmpz_factor_t value;
    FlintFactors() { fmpz_factor_init(value); }
    ~FlintFactors() { fmpz_factor_clear(value); }
    FlintFactors(const FlintFactors&) = delete;
};
inline bool proven_prime(const mpz_class& p) {
    if (p < 2) return false;
    FlintInteger number(p);
    return fmpz_is_prime(number.value) == 1;
}

/** Return all distinct prime divisors, verifying their proofs and product.
 * No partial factorization or probable-prime status crosses this boundary.
 * FLINT's complete routine has no bounded-runtime guarantee.
 */
inline std::vector<mpz_class> factor_proved(const mpz_class& positive) {
    if (positive < 1) throw std::logic_error("principal minor must be nonzero");
    if (positive == 1) return {};
    FlintInteger number(positive), product(1);
    FlintFactors factors;
    fmpz_factor(factors.value, number.value);
    fmpz_factor_expand(product.value, factors.value);
    if (!fmpz_equal(product.value, number.value))
        throw std::runtime_error("FLINT factorization is incomplete");
    std::vector<mpz_class> result;
    for (slong i = 0; i < factors.value->num; ++i) {
        if (fmpz_is_prime(factors.value->p + i) != 1)
            throw std::runtime_error("FLINT could not prove a factor prime");
        mpz_class p;
        fmpz_get_mpz(p.get_mpz_t(), factors.value->p + i);
        result.push_back(std::move(p));
    }
    return result;
}

/** Per-call memoization reuses complete factorizations of repeated |Delta_j|.
 * Previously proved factors are divided out before asking FLINT for new ones.
 */
inline std::vector<mpz_class> discover_primes(const std::vector<std::string>& minors,
                                             const Progress& progress) {
    std::set<mpz_class> known, completed;
    for (std::size_t j = 0; j < minors.size(); ++j) {
        report(progress, "factorization", j, minors.size());
        mpz_class n = abs(mpz_class(minors[j]));
        if (n == 0) throw std::logic_error("zero augmented principal minor");
        if (!completed.insert(n).second) continue;
        for (const auto& p : known) {
            while (mpz_divisible_p(n.get_mpz_t(), p.get_mpz_t())) {
                mpz_divexact(n.get_mpz_t(), n.get_mpz_t(), p.get_mpz_t());
                if (n == 1) break;
            }
            if (n == 1) break;
        }
        for (auto& p : factor_proved(n)) known.insert(std::move(p));
    }
    report(progress, "factorization", minors.size(), minors.size(), true);
    return {known.begin(), known.end()};
}

enum class Phase { rational, exhaustive, complete, working, failed };
inline std::string phase_name(Phase phase) {
    switch (phase) {
        case Phase::rational: return "rational";
        case Phase::exhaustive: return "exhaustive";
        case Phase::complete: return "complete";
        case Phase::working: return "working";
        default: return "failed";
    }
}

/** One explicit mutable state for Algorithms 1--5. Completion discards all
 * large algebraic workspace and retains only profiles, primes and identity.
 */
struct ReductionState {
    int rows = 0, cols = 0;
    std::uint64_t identity = 0;
    Phase phase = Phase::working;
    std::vector<int> generic_pivots;
    std::vector<mpz_class> primes;
    std::map<mpz_class, std::vector<int>> exceptional_pivots;
    std::map<std::string, double> timings;
    std::shared_ptr<NativeEngine> engine;
    void require(Phase expected) const {
        if (phase != expected)
            throw std::logic_error("invalid reduction stage: state is " + phase_name(phase));
    }
    void fail() { phase = Phase::failed; engine.reset(); exceptional_pivots.clear(); }
};

struct EnginePollGuard {
    std::shared_ptr<NativeEngine> engine;
    EnginePollGuard(std::shared_ptr<NativeEngine> ptr, const Progress& progress,
                    std::string phase, std::size_t total) : engine(std::move(ptr)) {
        engine->set_poll([progress, phase = std::move(phase), total]() {
            report(progress, phase, 0, total);
        });
    }
    ~EnginePollGuard() { engine->set_poll({}); }
};

inline std::shared_ptr<ReductionState> rational_reduction(Matrix& matrix, const Progress& progress) {
    matrix.require_available();
    auto state = std::make_shared<ReductionState>();
    state->rows = matrix.rows; state->cols = matrix.cols; state->identity = matrix.identity;
    auto start = Clock::now();
    report(progress, "rational_reduction", 0, matrix.cols, true);
    // Poll captures no Python object. The callback exists only during this call;
    // the engine callback is replaced at every continuation stage.
    state->engine = std::make_shared<LazyCompactDenseEngine>(matrix.rows, matrix.take(), [&]() {
        report(progress, "rational_reduction", 0, matrix.cols);
    });
    try {
        {
            EnginePollGuard poll(state->engine, progress, "rational_reduction", state->cols);
            state->engine->reduce();
        }
        state->timings["rational_reduction"] = elapsed(start);
        start = Clock::now();
        state->generic_pivots = state->engine->pivots();
        state->timings["profile_packaging_and_cleanup"] = elapsed(start);
        report(progress, "rational_reduction", matrix.cols, matrix.cols, true);
        start = Clock::now();
        report(progress, "factorization", 0, matrix.cols, true);
        {
            EnginePollGuard poll(state->engine, progress, "factorization", state->cols);
            state->primes = discover_primes(state->engine->minors(), progress);
        }
        state->timings["factorization"] = elapsed(start);
        state->phase = Phase::rational;
        return state;
    } catch (...) { state->fail(); throw; }
}

inline void exhaustive_rational_reduction(ReductionState& state, const Progress& progress) {
    state.require(Phase::rational);
    state.phase = Phase::working;
    const auto start = Clock::now();
    try {
        report(progress, "exhaustive_rational_reduction", 0, state.cols, true);
        EnginePollGuard poll(state.engine, progress, "exhaustive_rational_reduction", state.cols);
        if (!state.primes.empty()) state.engine->exhaustive();
        state.timings["exhaustive_rational_reduction"] = state.primes.empty() ? 0.0 : elapsed(start);
        report(progress, "exhaustive_rational_reduction", state.cols, state.cols, true);
        state.phase = Phase::exhaustive;
    } catch (...) { state.fail(); throw; }
}

inline void exceptional_correction(ReductionState& state, const Progress& progress) {
    state.require(Phase::exhaustive);
    state.phase = Phase::working;
    const auto start = Clock::now();
    try {
        report(progress, "exceptional_correction", 0, state.cols, true);
        GroupResult result;
        {
            EnginePollGuard poll(state.engine, progress, "exceptional_correction", state.cols);
            if (!state.primes.empty()) {
                std::vector<std::string> encoded;
                for (const auto& p : state.primes) encoded.push_back(p.get_str());
                result = state.engine->correct_groups(encoded);
            }
        }
        state.timings["exceptional_correction"] = state.primes.empty() ? 0.0 : elapsed(start);
        const auto output_started = Clock::now();
        for (const auto& group : result.groups) {
            auto pivots = group.state->pivots();
            for (const auto& p : group.primes) state.exceptional_pivots.emplace(mpz_class(p), pivots);
        }
        result.groups.clear();
        state.engine.reset();
        state.timings["profile_packaging_and_cleanup"] += elapsed(output_started);
        report(progress, "exceptional_correction", state.cols, state.cols, true);
        state.phase = Phase::complete;
    } catch (...) { state.fail(); throw; }
}

}  // namespace integral_native
