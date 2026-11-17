#pragma once

#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace integral_native {

using OrdinaryPair = std::pair<int, int>;
using OrdinaryPairs = std::vector<OrdinaryPair>;

/** Validate full boundary pivots before ordinary persistence extraction.
 *
 * Each sequence represents an N-by-N boundary with dummy rows 0,...,N-1
 * followed by original rows N,...,2N-1. A dummy pivot must be its own column;
 * an original pivot pairs an earlier zero reduced boundary with a coface
 * dimension. No grades or field-independence certificates are involved.
 *
 * This comparison helper returns only source indices. The public native
 * decoder handles grades separately, so these indices can be read at either
 * simplexwise or original filtration times without repeating reduction.
 */
inline void validate_ordinary_pairs_input(
    const std::vector<std::vector<int>>& sequences,
    const std::vector<int>& dimensions, int degree) {
  if (degree < 0) throw std::invalid_argument("homology degree must be nonnegative");
  if (dimensions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
    throw std::overflow_error("augmented boundary indices exceed the signed 32-bit range");
  for (int dimension : dimensions)
    if (dimension < 0) throw std::invalid_argument("dimensions must be nonnegative");
  const int size = static_cast<int>(dimensions.size());
  for (const auto& sequence : sequences) {
    if (sequence.size() != dimensions.size())
      throw std::invalid_argument("one distinct augmented pivot is required per column");
    std::vector<bool> seen(2 * size, false);
    for (int column = 0; column < size; ++column) {
      const int pivot = sequence[column];
      if (pivot < 0 || pivot >= 2 * size)
        throw std::invalid_argument("pivot outside the augmented matrix");
      if (seen[pivot])
        throw std::invalid_argument("one distinct augmented pivot is required per column");
      seen[pivot] = true;
      if (pivot < size && pivot != column)
        throw std::invalid_argument("a canonical dummy pivot must equal its column index");
    }
    for (int column = 0; column < size; ++column) {
      const int pivot = sequence[column];
      if (pivot < size) continue;
      const int row = pivot - size;
      // Subtraction is safe after the ordering test, including private calls
      // that supply dimensions beyond those possible for this filtration.
      if (row >= column || dimensions[column] <= dimensions[row] ||
          dimensions[column] - dimensions[row] != 1)
        throw std::invalid_argument("pivots are inconsistent with a boundary reduction");
      if (sequence[row] != row)
        throw std::invalid_argument("a paired birth must have zero reduced boundary");
    }
  }
}

/** Extract all requested pairs after validate_ordinary_pairs_input succeeds.
 *
 * Building the death map and allocating the returned pair vectors are part
 * of extraction. No validation or language-boundary conversion belongs to
 * this helper, and repeated sequences still produce separate outputs.
 */
inline std::vector<OrdinaryPairs> extract_validated_ordinary_pairs(
    const std::vector<std::vector<int>>& sequences,
    const std::vector<int>& dimensions, int degree) {
  const int size = static_cast<int>(dimensions.size());
  std::vector<OrdinaryPairs> result;
  result.reserve(sequences.size());
  for (const auto& sequence : sequences) {
    std::vector<int> deaths(size, -1);
    for (int column = 0; column < size; ++column) {
      const int pivot = sequence[column];
      if (pivot >= size) deaths[pivot - size] = column;
    }
    OrdinaryPairs pairs;
    for (int row = 0; row < size; ++row)
      if (dimensions[row] == degree && sequence[row] == row)
        pairs.emplace_back(row, deaths[row]);
    result.push_back(std::move(pairs));
  }
  return result;
}

/** Safely extract ordinary simplex-index pairs, with essential death -1. */
inline std::vector<OrdinaryPairs> ordinary_pairs(
    const std::vector<std::vector<int>>& sequences,
    const std::vector<int>& dimensions, int degree) {
  validate_ordinary_pairs_input(sequences, dimensions, degree);
  return extract_validated_ordinary_pairs(sequences, dimensions, degree);
}

/** Return ordinary pairs and native extraction seconds, excluding validation.
 *
 * Validation finishes before the steady-clock measurement begins. The timer
 * includes death-map construction and pair-vector allocation, and stops
 * before any Python return-value conversion or filtration-grade decoding.
 */
inline std::pair<std::vector<OrdinaryPairs>, double> ordinary_pairs_timed(
    const std::vector<std::vector<int>>& sequences,
    const std::vector<int>& dimensions, int degree) {
  validate_ordinary_pairs_input(sequences, dimensions, degree);
  const auto start = std::chrono::steady_clock::now();
  auto pairs = extract_validated_ordinary_pairs(sequences, dimensions, degree);
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  return {std::move(pairs), seconds};
}

}  // namespace integral_native
