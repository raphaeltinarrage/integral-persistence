#pragma once

#include "integer_column.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace integral_native {

/** Primitive common-denominator columns with 16-byte machine-word entries.
 *
 * A compact column represents A/d with signed 64-bit numerators and positive
 * signed 64-bit d. Pivot cancellation uses checked 128-bit intermediates in
 * a separate result buffer. If any numerator or denominator would leave the
 * compact range, the untouched original column is promoted to IntegerColumn
 * and the entire operation is recomputed exactly with GMP. Normalization
 * divides numerators and denominator together, never rescaling the column.
 */
class CompactColumn {
 public:
  using Entry = std::pair<int, mpq_class>;
  using Entries = std::vector<Entry>;
  using SmallEntry = std::pair<int, std::int64_t>;
  using Wide = __int128_t;

  CompactColumn() = default;
  /** Consume integer coefficients, retaining GMP storage only when needed. */
  static CompactColumn from_integers(std::vector<IntegerColumn::IntegerEntry> input) {
    std::sort(input.begin(), input.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    CompactColumn result;
    result.numerators_.reserve(input.size());
    int previous = -1;
    bool compact = true;
    for (const auto& [row, value] : input) {
      if (row <= previous || value == 0)
        throw std::invalid_argument("integer entries require distinct nonnegative rows and nonzero values");
      previous = row;
      std::int64_t word;
      if (compact && signed_word(value.get_mpz_t(), word))
        result.numerators_.emplace_back(row, word);
      else compact = false;
    }
    if (!compact) {
      result.numerators_.clear();
      result.numerators_.shrink_to_fit();
      result.large_ = std::make_unique<IntegerColumn>(IntegerColumn::from_integers(std::move(input)));
    }
    return result;
  }

  /** Adopt sorted machine-word entries, normalizing their shared denominator.
   *
   * This is a validated representation boundary for reusable accumulators.
   * Dividing all numerators and the denominator by their common divisor
   * preserves the exact column, including its scale and signs.
   */
  static CompactColumn from_sorted_small(std::vector<SmallEntry> entries,
                                         std::int64_t denominator) {
    if (denominator <= 0) throw std::invalid_argument("column denominator must be positive");
    int previous = -1;
    std::uint64_t common = static_cast<std::uint64_t>(denominator);
    for (const auto& [row, numerator] : entries) {
      if (row <= previous || numerator == 0)
        throw std::invalid_argument("compact entries must have sorted distinct nonnegative rows and nonzero values");
      previous = row;
      if (common > 1) common = std::gcd(common, magnitude(numerator));
    }
    CompactColumn result;
    if (entries.empty()) return result;
    if (common > 1) {
      const auto divisor = static_cast<std::int64_t>(common);
      for (auto& entry : entries) entry.second /= divisor;
      denominator /= divisor;
    }
    result.numerators_ = std::move(entries);
    result.denominator_ = denominator;
    return result;
  }
  explicit CompactColumn(Entries input) {
    std::sort(input.begin(), input.end(),
              [](const Entry& a, const Entry& b) { return a.first < b.first; });
    // Simplicial boundary and augmented-identity entries take this path,
    // without constructing per-column GMP denominators or computing lcms.
    bool integral = true;
    numerators_.reserve(input.size());
    int previous = -1;
    for (const auto& [row, value] : input) {
      if (row < 0 || row == previous)
        throw std::invalid_argument("column row IDs must be nonnegative and distinct");
      previous = row;
      if (mpz_sgn(mpq_denref(value.get_mpq_t())) == 0)
        throw std::invalid_argument("a rational coefficient has denominator zero");
      if (!value) throw std::invalid_argument("column entries must be nonzero");
      std::int64_t integer;
      if (integral && mpz_cmp_ui(mpq_denref(value.get_mpq_t()), 1) == 0 &&
          signed_word(mpq_numref(value.get_mpq_t()), integer)) {
        numerators_.emplace_back(row, integer);
      } else {
        integral = false;
      }
    }
    if (!integral) {
      large_ = std::make_unique<IntegerColumn>(std::move(input));
      numerators_.clear();
      try_demote();
    }
  }
  CompactColumn(const CompactColumn& other)
      : numerators_(other.numerators_), denominator_(other.denominator_),
        large_(other.large_ ? std::make_unique<IntegerColumn>(*other.large_) : nullptr) {}
  CompactColumn(CompactColumn&&) noexcept = default;
  CompactColumn& operator=(CompactColumn&&) noexcept = default;
  CompactColumn& operator=(const CompactColumn& other) {
    if (this != &other) {
      numerators_ = other.numerators_;
      denominator_ = other.denominator_;
      large_ = other.large_ ? std::make_unique<IntegerColumn>(*other.large_) : nullptr;
    }
    return *this;
  }

  std::size_t size() const noexcept { return large_ ? large_->size() : numerators_.size(); }
  bool empty() const noexcept { return size() == 0; }
  bool is_compact() const noexcept { return !large_; }

  const std::vector<SmallEntry>& small_entries() const {
    if (large_) throw std::logic_error("promoted column has no machine-word entry view");
    return numerators_;
  }
  std::int64_t small_denominator() const {
    if (large_) throw std::logic_error("promoted column has no machine-word denominator");
    return denominator_;
  }

  /** Read compact storage without constructing an intermediate GMP value.
   *
   * Return false for promoted storage. A missing row returns zero. The
   * positive shared denominator need not be reduced against this numerator;
   * callers constructing a scalar normalize the two words together.
   */
  bool small_coefficient(int row, std::int64_t& numerator,
                         std::int64_t& denominator) const noexcept {
    if (large_) return false;
    const auto* value = find_numerator(row);
    numerator = value ? *value : 0;
    denominator = denominator_;
    return true;
  }

  mpq_class coefficient(int row) const {
    if (large_) return large_->coefficient(row);
    const auto* numerator = find_numerator(row);
    return numerator ? rational(*numerator, denominator_) : mpq_class(0);
  }
  Entries entries() const {
    if (large_) return large_->entries();
    Entries result;
    result.reserve(numerators_.size());
    for (const auto& [row, numerator] : numerators_)
      result.emplace_back(row, rational(numerator, denominator_));
    return result;
  }
  int canonical_pivot() const {
    if (large_) return large_->canonical_pivot();
    if (empty()) throw std::logic_error("augmented column vanished");
    return numerators_.back().first;
  }
  int pivot(const std::vector<int>& positions) const {
    if (large_) return large_->pivot(positions);
    int best = canonical_pivot();
    for (const auto& entry : numerators_)
      if (positions[entry.first] < positions[best]) best = entry.first;
    return best;
  }
  int modular_pivot(const mpz_class& prime, bool debug) const {
    const int row = modular_pivot_or(prime, -1, debug);
    if (row < 0) throw std::logic_error("augmented column vanished modulo prime");
    return row;
  }

  /** Highest surviving original row, or the supplied dummy-row fallback.
   *
   * Lazy engines store only D*V here: it may vanish modulo p while the full
   * augmented column still has its unit diagonal entry in V. Unsafe
   * denominators remain errors in debug mode, including promoted storage.
   */
  int modular_pivot_or(const mpz_class& prime, int fallback, bool debug) const {
    if (prime < 2) throw std::invalid_argument("a modular pivot requires p >= 2");
    if (large_) {
      if (debug && mpz_divisible_p(large_->denominator().get_mpz_t(), prime.get_mpz_t()))
        throw std::logic_error("unsafe denominator in the current column");
      const auto& entries = large_->integer_entries();
      if (mpz_fits_ulong_p(prime.get_mpz_t())) {
        const auto small = prime.get_ui();
        for (auto it = entries.rbegin(); it != entries.rend(); ++it)
          if (!mpz_divisible_ui_p(it->second.get_mpz_t(), small)) return it->first;
      } else {
        for (auto it = entries.rbegin(); it != entries.rend(); ++it)
          if (!mpz_divisible_p(it->second.get_mpz_t(), prime.get_mpz_t())) return it->first;
      }
      return fallback;
    }
    if (empty()) return fallback;
    if (mpz_sizeinbase(prime.get_mpz_t(), 2) > 64) return canonical_pivot();
    std::uint64_t modulus = 0;
    std::size_t count = 0;
    mpz_export(&modulus, &count, -1, sizeof(modulus), 0, 0, prime.get_mpz_t());
    if (debug && static_cast<std::uint64_t>(denominator_) % modulus == 0)
      throw std::logic_error("unsafe denominator in the current column");
    for (auto it = numerators_.rbegin(); it != numerators_.rend(); ++it)
      if (magnitude(it->second) % modulus != 0) return it->first;
    return fallback;
  }

  /** Generic exact operation; pivot cancellation has a separate fast path. */
  void subtract(const CompactColumn& source, const mpq_class& scalar) {
    if (this == &source) throw std::logic_error("self cancellation is forbidden");
    if (!scalar || source.empty()) return;
    promote();
    if (source.large_) {
      large_->subtract(*source.large_, scalar);
    } else {
      const IntegerColumn lifted_source(source.entries());
      large_->subtract(lifted_source, scalar);
    }
    try_demote();
  }

  /** Cancel row r: ((B_r/g) A - (A_r/g) B)/(d B_r/g).
   *
   * The source denominator cancels algebraically. g=gcd(|A_r|,|B_r|)
   * avoids unnecessarily large products. A positive denominator is enforced
   * by changing both scales' signs. All compact operands remain untouched
   * until every merged entry and the primitive denominator have been checked.
   */
  void cancel_at(const CompactColumn& source, int row) {
    if (this == &source) throw std::logic_error("self cancellation is forbidden");
    if (large_ || source.large_) {
      fallback_cancel(source, row);
      return;
    }
    const auto* target_value = find_numerator(row);
    const auto* source_value = source.find_numerator(row);
    if (!target_value || !source_value)
      throw std::logic_error("cancellation row must be nonzero in both columns");
    const std::uint64_t divisor = std::gcd(magnitude(*target_value), magnitude(*source_value));
    Wide target_scale = static_cast<Wide>(*source_value) / static_cast<Wide>(divisor);
    Wide source_scale = static_cast<Wide>(*target_value) / static_cast<Wide>(divisor);
    if (target_scale < 0) {
      target_scale = -target_scale;
      source_scale = -source_scale;
    }
    Wide wide_denominator;
    if (__builtin_mul_overflow(static_cast<Wide>(denominator_), target_scale, &wide_denominator) ||
        !positive_word(wide_denominator)) {
      fallback_cancel(source, row);
      return;
    }
    const auto denominator = static_cast<std::int64_t>(wide_denominator);
    std::uint64_t common = static_cast<std::uint64_t>(denominator);
    std::vector<SmallEntry> result;
    result.reserve(numerators_.size() + source.numerators_.size());
    const bool target_unit = target_scale == 1;
    const int source_unit = source_scale == 1 ? 1 : (source_scale == -1 ? -1 : 0);
    auto append = [&](int index, std::int64_t target, std::int64_t addend) {
      std::int64_t integer;
      if (target_unit && source_unit == 1) {
        if (__builtin_sub_overflow(target, addend, &integer)) return false;
      } else if (target_unit && source_unit == -1) {
        if (__builtin_add_overflow(target, addend, &integer)) return false;
      } else if (target_unit && addend == 0) {
        integer = target;
      } else {
        Wide left, right, value;
        if (__builtin_mul_overflow(target_scale, static_cast<Wide>(target), &left) ||
            __builtin_mul_overflow(source_scale, static_cast<Wide>(addend), &right) ||
            __builtin_sub_overflow(left, right, &value) || !signed_word(value))
          return false;
        integer = static_cast<std::int64_t>(value);
      }
      if (integer != 0) {
        result.emplace_back(index, integer);
        if (common > 1) common = std::gcd(common, magnitude(integer));
      }
      return true;
    };
    std::size_t i = 0, j = 0;
    while (i < numerators_.size() || j < source.numerators_.size()) {
      bool fits;
      if (j == source.numerators_.size() ||
          (i < numerators_.size() && numerators_[i].first < source.numerators_[j].first)) {
        fits = append(numerators_[i].first, numerators_[i].second, 0);
        ++i;
      } else if (i == numerators_.size() || source.numerators_[j].first < numerators_[i].first) {
        fits = append(source.numerators_[j].first, 0, source.numerators_[j].second);
        ++j;
      } else {
        fits = append(numerators_[i].first, numerators_[i].second, source.numerators_[j].second);
        ++i;
        ++j;
      }
      if (!fits) {
        fallback_cancel(source, row);
        return;
      }
    }
    if (result.empty()) {
      denominator_ = 1;
    } else if (common > 1) {
      // common divides a positive signed-word denominator, so conversion and
      // signed division are safe even for an INT64_MIN numerator.
      const auto signed_common = static_cast<std::int64_t>(common);
      for (auto& entry : result) entry.second /= signed_common;
      denominator_ = denominator / signed_common;
    } else {
      denominator_ = denominator;
    }
    numerators_.swap(result);
  }

 private:
  std::vector<SmallEntry> numerators_;
  std::int64_t denominator_ = 1;
  std::unique_ptr<IntegerColumn> large_;

  static std::uint64_t magnitude(std::int64_t value) noexcept {
    return value < 0 ? std::uint64_t(-(value + 1)) + 1 : std::uint64_t(value);
  }
  static bool signed_word(Wide value) noexcept {
    return value >= std::numeric_limits<std::int64_t>::min() &&
           value <= std::numeric_limits<std::int64_t>::max();
  }
  static bool positive_word(Wide value) noexcept {
    return value > 0 && value <= std::numeric_limits<std::int64_t>::max();
  }
  static bool signed_word(mpz_srcptr value, std::int64_t& output) {
    if constexpr (sizeof(long) == sizeof(std::int64_t)) {
      if (!mpz_fits_slong_p(value)) return false;
      output = static_cast<std::int64_t>(mpz_get_si(value));
      return true;
    }
    if (mpz_sizeinbase(value, 2) > 64) return false;
    std::uint64_t absolute = 0;
    std::size_t count = 0;
    mpz_export(&absolute, &count, -1, sizeof(absolute), 0, 0, value);
    constexpr auto maximum = std::uint64_t(std::numeric_limits<std::int64_t>::max());
    if (mpz_sgn(value) >= 0) {
      if (absolute > maximum) return false;
      output = static_cast<std::int64_t>(absolute);
    } else {
      if (absolute > maximum + 1) return false;
      output = absolute == maximum + 1 ? std::numeric_limits<std::int64_t>::min()
                                      : -static_cast<std::int64_t>(absolute);
    }
    return true;
  }
  static mpq_class rational(std::int64_t numerator, std::int64_t denominator) {
    mpq_class result;
    if constexpr (sizeof(long) >= sizeof(std::int64_t)) {
      mpq_set_si(result.get_mpq_t(), static_cast<long>(numerator),
                 static_cast<unsigned long>(denominator));
    } else {
      const auto absolute = magnitude(numerator);
      const auto unsigned_denominator = static_cast<std::uint64_t>(denominator);
      mpz_import(mpq_numref(result.get_mpq_t()), 1, -1, sizeof(absolute), 0, 0, &absolute);
      mpz_import(mpq_denref(result.get_mpq_t()), 1, -1, sizeof(unsigned_denominator), 0, 0,
                 &unsigned_denominator);
      if (numerator < 0) mpz_neg(mpq_numref(result.get_mpq_t()), mpq_numref(result.get_mpq_t()));
    }
    result.canonicalize();
    return result;
  }
  const std::int64_t* find_numerator(int row) const {
    const auto found = std::lower_bound(numerators_.begin(), numerators_.end(), row,
        [](const SmallEntry& entry, int key) { return entry.first < key; });
    return found != numerators_.end() && found->first == row ? &found->second : nullptr;
  }
  void promote() {
    if (large_) return;
    auto lifted = std::make_unique<IntegerColumn>(entries());
    large_ = std::move(lifted);
    std::vector<SmallEntry>().swap(numerators_);
    denominator_ = 1;
  }
  void try_demote() {
    if (!large_) return;
    std::int64_t denominator;
    if (!signed_word(large_->denominator().get_mpz_t(), denominator) || denominator <= 0) return;
    std::vector<SmallEntry> packed;
    packed.reserve(large_->size());
    for (const auto& [row, numerator] : large_->integer_entries()) {
      std::int64_t value;
      if (!signed_word(numerator.get_mpz_t(), value)) return;
      packed.emplace_back(row, value);
    }
    denominator_ = denominator;
    numerators_.swap(packed);
    large_.reset();
  }
  void fallback_cancel(const CompactColumn& source, int row) {
    promote();
    if (source.large_) {
      large_->cancel_at(*source.large_, row);
    } else {
      const IntegerColumn lifted_source(source.entries());
      large_->cancel_at(lifted_source, row);
    }
    try_demote();
  }
};

static_assert(sizeof(CompactColumn::SmallEntry) == 16,
              "compact column performance requires 16-byte row/numerator entries");

}  // namespace integral_native
