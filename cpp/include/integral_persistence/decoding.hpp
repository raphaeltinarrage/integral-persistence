#pragma once

#include "reduction.hpp"
#include <algorithm>

namespace integral_native {
using IndexPair = std::pair<int, int>;  // source birth, source death (-1 = infinity)
using IndexPairs = std::vector<IndexPair>;

/** Source-index intervals before regrading. No simultaneous interval is lost
 * during construction or reduction; Algorithm 6 removes it only on decoding.
 */
inline IndexPairs index_pairs(const std::vector<int>& pivots, const MatrixData& data) {
    const int m = data.matrix->cols, n = data.matrix->rows;
    if (static_cast<int>(pivots.size()) != m) throw std::logic_error("profile shape mismatch");
    std::vector<int> deaths(n, -1);
    for (int col = 0; col < m; ++col) {
        const int pivot = pivots[col];
        if (pivot < 0 || pivot >= n + m) throw std::logic_error("invalid augmented pivot");
        if (pivot < m) {
            if (pivot != col) throw std::logic_error("invalid canonical dummy pivot");
        } else {
            const int row = pivot - m;
            if (deaths[row] != -1) throw std::logic_error("repeated pivot in profile");
            deaths[row] = col;
        }
    }
    IndexPairs result;
    if (data.module == "field") {
        if (!data.field || n != m || data.field->dimensions.size() != pivots.size() || data.field->simplex_sources.size() != pivots.size())
            throw std::logic_error("boundary metadata shape mismatch");
        for (int row = 0; row < n; ++row) {
            const int death = deaths[row];
            if (death != -1 && (death <= row || pivots[row] != row ||
                data.field->dimensions[death] != data.field->dimensions[row] + 1))
                throw std::logic_error("profile is inconsistent with a boundary matrix");
            if (data.field->dimensions[row] == data.degree && pivots[row] == row)
                result.emplace_back(data.field->simplex_sources[row], death == -1 ? -1 : data.field->simplex_sources[death]);
        }
    } else {
        if (data.birth_sources.size() != static_cast<std::size_t>(n) ||
            data.relation_sources.size() != static_cast<std::size_t>(m))
            throw std::logic_error("presentation metadata shape mismatch");
        for (int row = 0; row < n; ++row)
            result.emplace_back(data.birth_sources[row], deaths[row] == -1 ? -1 : data.relation_sources[deaths[row]]);
    }
    return result;
}

inline int compare_deaths(int a, int b, const Grades& times) {
    if (a < 0) return b < 0 ? 0 : 1;
    if (b < 0) return -1;
    return compare(times.at(a), times.at(b));
}
inline IndexPairs grade_pairs(IndexPairs pairs, const Grades& times) {
    IndexPairs result;
    result.reserve(pairs.size());
    for (const auto& [birth, death] : pairs) {
        if (birth < 0 || static_cast<std::size_t>(birth) >= times.size() || death < -1 ||
            (death >= 0 && static_cast<std::size_t>(death) >= times.size()))
            throw std::logic_error("source index outside grading");
        if (death >= 0) {
            const int order = compare(times[birth], times[death]);
            if (order > 0) throw std::logic_error("interval dies before it is born");
            if (order == 0) continue;
        }
        result.emplace_back(birth, death);
    }
    std::stable_sort(result.begin(), result.end(), [&](const IndexPair& a, const IndexPair& b) {
        const int births = compare(times[a.first], times[b.first]);
        return births < 0 || (births == 0 && compare_deaths(a.second, b.second, times) < 0);
    });
    return result;
}
inline bool equal_barcodes(const IndexPairs& a, const IndexPairs& b, const Grades& times) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (compare(times[a[i].first], times[b[i].first]) != 0 ||
            compare_deaths(a[i].second, b[i].second, times) != 0) return false;
    return true;
}

struct BarcodeResult {
    std::shared_ptr<const Grades> times;
    IndexPairs generic;
    std::map<mpz_class, IndexPairs> exceptional;
    std::vector<mpz_class> matrix_primes;
    std::string module;
    int degree;
    std::map<std::string, double> timings;
};

/** Algorithm 6: decode the completed profiles at a nondecreasing regrading.
 * The matrix identity prevents applying unrelated construction metadata. All
 * comparisons preserve interval multiplicities and exact integer/float ties.
 */
inline std::shared_ptr<BarcodeResult> decode_barcodes(const ReductionState& state,
                                                     const MatrixData& data,
                                                     std::shared_ptr<const Grades> times) {
    state.require(Phase::complete);
    if (state.identity != data.matrix->identity)
        throw std::invalid_argument("decoding metadata belongs to a different matrix");
    validate_grades(*times, data.source_count);
    auto result = std::make_shared<BarcodeResult>();
    result->times = std::move(times);
    result->module = data.module; result->degree = data.degree;
    result->matrix_primes = state.primes;
    const auto start = Clock::now();
    result->generic = grade_pairs(index_pairs(state.generic_pivots, data), *result->times);
    for (const auto& [prime, profile] : state.exceptional_pivots) {
        auto barcode = grade_pairs(index_pairs(profile, data), *result->times);
        if (!equal_barcodes(result->generic, barcode, *result->times))
            result->exceptional.emplace(prime, std::move(barcode));
    }
    result->timings["decode_barcodes"] = elapsed(start);
    return result;
}
}  // namespace integral_native
