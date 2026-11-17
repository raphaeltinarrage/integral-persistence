#pragma once

#include "hybrid_column.hpp"

#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace integral_native {

/** Exact integer cancellation multipliers and the original rational factor.
 *
 * For target A/d and source B/e at row r, let g=gcd(|A_r|,|B_r|),
 * a=|B_r|/g and b=sign(B_r)*A_r/g. Then target becomes (a*A-b*B)/(a*d),
 * while the transformation must receive the factor b*e/(a*d). The positive
 * target multiplier preserves signs without ever taking abs(INT64_MIN).
 */
struct CompactCancellation {
  using Wide = __int128_t;
  Wide target_scale;
  Wide source_scale;
  HybridRational factor;
};

inline CompactCancellation compact_cancellation(std::int64_t target_numerator,
                                                 std::int64_t target_denominator,
                                                 std::int64_t source_numerator,
                                                 std::int64_t source_denominator) {
  if (!target_numerator || !source_numerator)
    throw std::domain_error("cancellation pivot numerators must be nonzero");
  if (target_denominator <= 0 || source_denominator <= 0)
    throw std::invalid_argument("cancellation denominators must be positive");
  using Wide = CompactCancellation::Wide;
  const auto magnitude = [](std::int64_t value) {
    return value < 0 ? std::uint64_t(-(value + 1)) + 1 : std::uint64_t(value);
  };
  const auto target_magnitude = magnitude(target_numerator);
  const auto source_magnitude = magnitude(source_numerator);
  const auto divisor = std::gcd(target_magnitude, source_magnitude);
  // Magnitudes, including 2**63 for INT64_MIN, fit uint64. Divide those
  // machine words before widening rather than invoking signed Wide division.
  const Wide a = static_cast<Wide>(source_magnitude / divisor);
  Wide b = static_cast<Wide>(target_magnitude / divisor);
  if ((target_numerator < 0) != (source_numerator < 0)) b = -b;
  // Both products have magnitude at most 2**126: the signed 128-bit
  // multiplications cannot overflow, even with a source pivot INT64_MIN.
  const Wide numerator = b * source_denominator;
  const Wide denominator = a * target_denominator;
  HybridRational factor;
  if (numerator >= std::numeric_limits<std::int64_t>::min() &&
      numerator <= std::numeric_limits<std::int64_t>::max() &&
      denominator <= std::numeric_limits<std::int64_t>::max()) {
    factor = HybridRational(static_cast<std::int64_t>(numerator),
                            static_cast<std::int64_t>(denominator));
  } else {
    factor = HybridRational::from_wide(numerator, denominator);
  }
  return {a, b, std::move(factor)};
}

}  // namespace integral_native
