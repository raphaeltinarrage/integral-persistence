#pragma once

#include <gmpxx.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace integral_native {

using MatrixColumn = std::vector<std::pair<int, mpz_class>>;
using IntegerColumns = std::vector<MatrixColumn>;
using Clock = std::chrono::steady_clock;
using Progress = std::function<void(const std::string&, std::size_t, std::size_t, bool)>;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
inline void report(const Progress& callback, const std::string& phase,
                   std::size_t done, std::size_t total, bool force = false) {
    if (callback) callback(phase, done, total, force);
}

/** A grade preserves its supplied integer or binary floating-point value.
 * Mixed comparison uses an exact binary rational, never an integer-to-double
 * conversion. Infinity is reserved for output deaths, not input grades.
 */
struct Grade {
    bool integral = true;
    mpz_class integer = 0;
    double floating = 0;
    explicit Grade(mpz_class value = 0) : integer(std::move(value)) {}
    explicit Grade(double value) : integral(false), floating(value) {
        if (!std::isfinite(value)) throw std::invalid_argument("filtration times must be finite");
    }
};
inline int compare(const Grade& a, const Grade& b) {
    if (a.integral && b.integral) return cmp(a.integer, b.integer);
    if (!a.integral && !b.integral)
        return (a.floating > b.floating) - (a.floating < b.floating);
    if (a.integral) return cmp(mpq_class(a.integer), mpq_class(b.floating));
    return cmp(mpq_class(a.floating), mpq_class(b.integer));
}
using Grades = std::vector<Grade>;
inline void validate_grades(const Grades& times, std::size_t size) {
    if (times.size() != size) throw std::invalid_argument("one time is required per source simplex");
    for (std::size_t i = 1; i < times.size(); ++i)
        if (compare(times[i - 1], times[i]) > 0)
            throw std::invalid_argument("times must be nondecreasing in the fixed simplex order");
}

/** Read-only validated matrix storage, transferred exactly once to reduction.
 * Every Python alias shares this handle. copy() is the explicit preservation
 * operation; the logical identity is retained so its metadata remains valid.
 */
struct Matrix {
    int rows;
    int cols;
    IntegerColumns columns;
    bool consumed = false;
    std::uint64_t identity;
    inline static std::atomic<std::uint64_t> next_identity{1};
    Matrix(int nrows, IntegerColumns values)
        : rows(nrows), cols(static_cast<int>(values.size())), columns(std::move(values)),
          identity(next_identity++) {
        if (nrows < 0 || columns.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() - nrows))
            throw std::invalid_argument("augmented matrix exceeds the supported index range");
    }
    void require_available() const {
        if (consumed) throw std::logic_error("matrix has been consumed; copy it before reduction to preserve it");
    }
    std::shared_ptr<Matrix> copy() const {
        require_available();
        auto result = std::make_shared<Matrix>(rows, columns);
        result->identity = identity;
        return result;
    }
    IntegerColumns take() {
        require_available();
        consumed = true;
        return std::move(columns);
    }
};

/** Internal common order. Simplices have sorted, remapped integer vertices;
 * every proper face precedes its cofaces. Numeric times are optional here.
 */
struct PreparedFiltration {
    std::vector<std::vector<int>> simplices;
    std::shared_ptr<const Grades> times;
    int vertex_count = 0;
};

/** Index-only decoding metadata survives consumption of the coefficient matrix.
 * Field matrices use simplex_sources/dimensions; presentations use the source
 * indices of generator births and relation births, including simultaneous ones.
 */
struct FieldMetadata {
    std::vector<int> dimensions, simplex_sources;
};
struct MatrixData {
    std::shared_ptr<Matrix> matrix;
    std::string module;
    int degree = 0;
    std::size_t source_count = 0;
    std::shared_ptr<const FieldMetadata> field;
    std::vector<int> birth_sources, relation_sources;
    std::shared_ptr<const Grades> times;
    std::map<std::string, double> timings;
};
using BuiltMatrices = std::map<int, std::map<std::string, std::shared_ptr<MatrixData>>>;

BuiltMatrices build_matrices(const PreparedFiltration& input,
                            const std::vector<int>& degrees,
                            const std::vector<std::string>& modules,
                            const Progress& progress = {});

}  // namespace integral_native
