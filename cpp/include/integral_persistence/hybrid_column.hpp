#pragma once

#include <gmpxx.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace integral_native {

/** Exact rational with an allocation-free signed 64-bit common case.
 *
 * A small value has a reduced signed numerator and a strictly positive
 * denominator. Checked signed 128-bit intermediates detect overflow before
 * it occurs. Values outside the small representation use an owned GMP
 * rational; every operation may return to the small representation after
 * exact cancellation. There is no saturation, rounding, or modular shortcut.
 */
class HybridRational {
 public:
  using Wide = __int128_t;
  using UnsignedWide = __uint128_t;

  HybridRational() = default;
  explicit HybridRational(const mpq_class& value) { assign(value); }
  /** Construct an exact small rational without a GMP allocation. */
  HybridRational(std::int64_t numerator, std::int64_t denominator) {
    if (denominator <= 0) throw std::invalid_argument("rational denominator must be positive");
    const auto divisor = static_cast<std::int64_t>(
        std::gcd(magnitude(numerator), static_cast<std::uint64_t>(denominator)));
    numerator_ = numerator / divisor;
    denominator_ = denominator / divisor;
  }
  HybridRational(const HybridRational& other)
      : numerator_(other.numerator_), denominator_(other.denominator_),
        large_(other.large_ ? std::make_unique<mpq_class>(*other.large_) : nullptr) {}
  HybridRational(HybridRational&&) noexcept = default;
  HybridRational& operator=(HybridRational&&) noexcept = default;
  HybridRational& operator=(const HybridRational& other) {
    if (this != &other) {
      numerator_ = other.numerator_;
      denominator_ = other.denominator_;
      large_ = other.large_ ? std::make_unique<mpq_class>(*other.large_) : nullptr;
    }
    return *this;
  }

  bool small() const noexcept { return !large_; }
  /** Read normalized machine words when the value has no GMP allocation. */
  bool small_parts(std::int64_t& numerator, std::int64_t& denominator) const noexcept {
    if (large_) return false;
    numerator = numerator_;
    denominator = denominator_;
    return true;
  }
  /** Internal fast-kernel commit; caller proves coprime words and d > 0. */
  void assign_reduced_small(std::int64_t numerator, std::int64_t denominator) noexcept {
    assert(denominator > 0);
    assert(std::gcd(magnitude(numerator), static_cast<std::uint64_t>(denominator)) == 1);
    numerator_ = numerator;
    denominator_ = denominator;
    large_.reset();
  }
  /** Normalize exact checked intermediate words, promoting to GMP if needed. */
  static HybridRational from_wide(Wide numerator, Wide denominator) {
    HybridRational result;
    result.assign_wide(numerator, denominator);
    return result;
  }
  bool is_zero() const noexcept { return large_ ? *large_ == 0 : numerator_ == 0; }
  bool is_one() const noexcept { return !large_ && numerator_ == 1 && denominator_ == 1; }
  bool is_minus_one() const noexcept { return !large_ && numerator_ == -1 && denominator_ == 1; }

  /** Exact quotient, avoiding GMP conversion when both operands are small. */
  static HybridRational quotient(const HybridRational& numerator,
                                  const HybridRational& denominator) {
    if (denominator.is_zero()) throw std::domain_error("zero rational divisor");
    HybridRational result;
    if (!numerator.large_ && !denominator.large_) {
      Wide top = static_cast<Wide>(numerator.numerator_) * denominator.denominator_;
      Wide bottom = static_cast<Wide>(numerator.denominator_) * denominator.numerator_;
      // Each product is at most 2**126 in magnitude, so sign normalization
      // cannot negate the minimum signed 128-bit value.
      if (bottom < 0) {
        top = -top;
        bottom = -bottom;
      }
      result.assign_wide(top, bottom);
    } else {
      result.assign(numerator.exact() / denominator.exact());
    }
    return result;
  }

  mpq_class exact() const {
    if (large_) return *large_;
    mpq_class result;
    if constexpr (sizeof(long) >= sizeof(std::int64_t)) {
      mpq_set_si(result.get_mpq_t(), static_cast<long>(numerator_),
                 static_cast<unsigned long>(denominator_));
    } else {
      const auto absolute = magnitude(numerator_);
      const auto denominator = static_cast<std::uint64_t>(denominator_);
      mpz_import(mpq_numref(result.get_mpq_t()), 1, -1, sizeof(absolute), 0, 0, &absolute);
      mpz_import(mpq_denref(result.get_mpq_t()), 1, -1, sizeof(denominator), 0, 0, &denominator);
      if (numerator_ < 0) mpz_neg(mpq_numref(result.get_mpq_t()), mpq_numref(result.get_mpq_t()));
    }
    return result;
  }

  /** Replace this by this - factor * source using checked exact arithmetic. */
  void subtract_product(const HybridRational& factor, const HybridRational& source) {
    if (factor.is_zero() || source.is_zero()) return;
    if (!large_ && !factor.large_ && !source.large_) {
      // Integral coefficients and unit cancellation dominate simplicial input.
      // A 64x64 product and one 64-bit subtraction fit a signed 128-bit value.
      if (denominator_ == 1 && factor.denominator_ == 1 && source.denominator_ == 1) {
        const Wide product = static_cast<Wide>(factor.numerator_) * source.numerator_;
        const Wide result = static_cast<Wide>(numerator_) - product;
        assign_wide(result, 1);
        return;
      }
      // Unit factors avoid a third rational multiplication and often expose
      // a common denominator directly (no wide gcd is needed in that case).
      if (factor.is_one() || factor.is_minus_one()) {
        if (denominator_ == source.denominator_) {
          assign_wide(static_cast<Wide>(numerator_) -
                      static_cast<Wide>(factor.numerator_) * source.numerator_, denominator_);
          return;
        }
      }
      Wide product_denominator, left, right, denominator, numerator;
      const Wide product_numerator = static_cast<Wide>(factor.numerator_) * source.numerator_;
      product_denominator = static_cast<Wide>(factor.denominator_) * source.denominator_;
      if (!__builtin_mul_overflow(static_cast<Wide>(numerator_), product_denominator, &left) &&
          !__builtin_mul_overflow(product_numerator, static_cast<Wide>(denominator_), &right) &&
          !__builtin_mul_overflow(static_cast<Wide>(denominator_), product_denominator, &denominator) &&
          !__builtin_sub_overflow(left, right, &numerator)) {
        assign_wide(numerator, denominator);
        return;
      }
    }
    // Compute from original operands if any checked intermediate overflows.
    // No partially updated value is used in the fallback.
    mpq_class value = exact();
    value -= factor.exact() * source.exact();
    assign(value);
  }

  bool numerator_divisible(const mpz_class& prime) const {
    if (large_) return mpz_divisible_p(mpq_numref(large_->get_mpq_t()), prime.get_mpz_t());
    return divisible_small(numerator_, prime);
  }
  bool denominator_divisible(const mpz_class& prime) const {
    if (large_) return mpz_divisible_p(mpq_denref(large_->get_mpq_t()), prime.get_mpz_t());
    return divisible_small(denominator_, prime);
  }

 private:
  std::int64_t numerator_ = 0;
  std::int64_t denominator_ = 1;
  std::unique_ptr<mpq_class> large_;

  static std::uint64_t magnitude(std::int64_t value) noexcept {
    return value < 0 ? std::uint64_t(-(value + 1)) + 1 : std::uint64_t(value);
  }
  static UnsignedWide magnitude(Wide value) noexcept {
    return value < 0 ? UnsignedWide(-(value + 1)) + 1 : UnsignedWide(value);
  }
  static mpz_class wide_integer(Wide value) {
    const auto absolute = magnitude(value);
    const std::uint64_t words[] = {std::uint64_t(absolute), std::uint64_t(absolute >> 64)};
    mpz_class result;
    mpz_import(result.get_mpz_t(), 2, -1, sizeof(words[0]), 0, 0, words);
    if (value < 0) mpz_neg(result.get_mpz_t(), result.get_mpz_t());
    return result;
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
  static bool divisible_small(std::int64_t value, const mpz_class& prime) {
    if (prime < 2) throw std::invalid_argument("characteristic must exceed one");
    if (value == 0) return true;
    const auto absolute = magnitude(value);
    if (mpz_sizeinbase(prime.get_mpz_t(), 2) > 64) return false;
    std::uint64_t modulus = 0;
    std::size_t count = 0;
    mpz_export(&modulus, &count, -1, sizeof(modulus), 0, 0, prime.get_mpz_t());
    return absolute % modulus == 0;
  }
  static UnsignedWide gcd(UnsignedWide left, UnsignedWide right) noexcept {
    while (right) {
      const auto remainder = left % right;
      left = right;
      right = remainder;
    }
    return left;
  }
  void assign_wide(Wide numerator, Wide denominator) {
    if (denominator <= 0) throw std::logic_error("nonpositive hybrid denominator");
    if (numerator == 0) {
      numerator_ = 0;
      denominator_ = 1;
      large_.reset();
      return;
    }
    if (denominator != 1) {
      // The positive denominator is at most signed-Wide::max, so the gcd
      // also fits signed Wide even for the most negative numerator.
      const auto divisor = static_cast<Wide>(gcd(magnitude(numerator), UnsignedWide(denominator)));
      numerator /= divisor;
      denominator /= divisor;
    }
    if (numerator >= std::numeric_limits<std::int64_t>::min() &&
        numerator <= std::numeric_limits<std::int64_t>::max() &&
        denominator <= std::numeric_limits<std::int64_t>::max()) {
      numerator_ = static_cast<std::int64_t>(numerator);
      denominator_ = static_cast<std::int64_t>(denominator);
      large_.reset();
    } else {
      large_ = std::make_unique<mpq_class>(wide_integer(numerator), wide_integer(denominator));
    }
  }
  void assign(const mpq_class& value) {
    if (mpz_sgn(mpq_denref(value.get_mpq_t())) == 0)
      throw std::invalid_argument("zero rational denominator");
    std::int64_t numerator, denominator;
    if (signed_word(mpq_numref(value.get_mpq_t()), numerator) &&
        signed_word(mpq_denref(value.get_mpq_t()), denominator) && denominator > 0) {
      assign_wide(numerator, denominator);
    } else {
      auto normalized = std::make_unique<mpq_class>(value);
      normalized->canonicalize();
      if (signed_word(mpq_numref(normalized->get_mpq_t()), numerator) &&
          signed_word(mpq_denref(normalized->get_mpq_t()), denominator) && denominator > 0) {
        assign_wide(numerator, denominator);
      } else {
        large_ = std::move(normalized);
      }
    }
  }
};

/** Contiguous sparse columns with exact small rationals and GMP fallback. */
class HybridColumn {
 public:
  using Entry = std::pair<int, HybridRational>;
  using ExportedEntry = std::pair<int, mpq_class>;

  explicit HybridColumn(std::vector<ExportedEntry> input) {
    std::sort(input.begin(), input.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    entries_.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (!input[i].second || (i && input[i - 1].first == input[i].first))
        throw std::invalid_argument("column entries must be distinct and nonzero");
      entries_.emplace_back(input[i].first, HybridRational(input[i].second));
    }
  }
  std::size_t size() const { return entries_.size(); }
  /** Adopt a finished accumulator's sorted support without coefficient conversion. */
  static HybridColumn from_sorted_hybrid(std::vector<Entry> entries) {
    int previous = -1;
    for (const auto& [row, value] : entries) {
      if (row <= previous || value.is_zero())
        throw std::invalid_argument("hybrid support must be sorted, distinct and nonzero");
      previous = row;
    }
    HybridColumn result(std::vector<ExportedEntry>{});
    result.entries_ = std::move(entries);
    return result;
  }
  const std::vector<Entry>& hybrid_entries() const noexcept { return entries_; }
  HybridRational coefficient_hybrid(int row) const {
    const auto found = std::lower_bound(entries_.begin(), entries_.end(), row,
                                       [](const Entry& entry, int key) { return entry.first < key; });
    return found != entries_.end() && found->first == row ? found->second : HybridRational();
  }
  std::vector<ExportedEntry> entries() const {
    std::vector<ExportedEntry> result;
    result.reserve(entries_.size());
    for (const auto& [row, value] : entries_) result.emplace_back(row, value.exact());
    return result;
  }
  mpq_class coefficient(int row) const {
    const auto found = std::lower_bound(entries_.begin(), entries_.end(), row,
                                       [](const Entry& entry, int key) { return entry.first < key; });
    return found != entries_.end() && found->first == row ? found->second.exact() : mpq_class(0);
  }
  int canonical_pivot() const {
    if (entries_.empty()) throw std::logic_error("augmented column vanished");
    return entries_.back().first;
  }
  int pivot(const std::vector<int>& positions) const {
    int row = canonical_pivot();
    for (const auto& entry : entries_)
      if (positions[entry.first] < positions[row]) row = entry.first;
    return row;
  }
  int modular_pivot(const mpz_class& prime, bool debug) const {
    if (prime < 2) throw std::invalid_argument("characteristic must exceed one");
    if (debug)
      for (const auto& entry : entries_)
        if (entry.second.denominator_divisible(prime))
          throw std::logic_error("unsafe denominator in current column");
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
      if (!it->second.numerator_divisible(prime)) return it->first;
    throw std::logic_error("augmented column vanished modulo prime");
  }
  void subtract(const HybridColumn& source, const mpq_class& scalar) {
    if (this == &source) throw std::logic_error("self cancellation is forbidden");
    if (!scalar) return;
    subtract_hybrid(source, HybridRational(scalar));
  }

  /** Cancel one physical row using native coefficients, preserving source. */
  void cancel_at(const HybridColumn& source, int row) {
    if (this == &source) throw std::logic_error("self cancellation is forbidden");
    const auto source_found = std::lower_bound(source.entries_.begin(), source.entries_.end(), row,
        [](const Entry& entry, int key) { return entry.first < key; });
    if (source_found == source.entries_.end() || source_found->first != row)
      throw std::domain_error("source coefficient is zero at cancellation row");
    const auto target_found = std::lower_bound(entries_.begin(), entries_.end(), row,
        [](const Entry& entry, int key) { return entry.first < key; });
    if (target_found == entries_.end() || target_found->first != row) return;
    const auto factor = HybridRational::quotient(target_found->second, source_found->second);
    subtract_hybrid(source, factor);
  }

  /** Native exact scalar update; a scalar copy also protects aliased inputs. */
  void subtract_hybrid(const HybridColumn& source, HybridRational factor) {
    if (this == &source) throw std::logic_error("self cancellation is forbidden");
    if (factor.is_zero() || source.entries_.empty()) return;
    std::vector<Entry> result;
    result.reserve(entries_.size() + source.entries_.size());
    std::size_t i = 0, j = 0;
    while (i < entries_.size() || j < source.entries_.size()) {
      if (j == source.entries_.size() ||
          (i < entries_.size() && entries_[i].first < source.entries_[j].first)) {
        result.push_back(std::move(entries_[i++]));
      } else if (i == entries_.size() || source.entries_[j].first < entries_[i].first) {
        HybridRational value;
        value.subtract_product(factor, source.entries_[j].second);
        result.emplace_back(source.entries_[j++].first, std::move(value));
      } else {
        entries_[i].second.subtract_product(factor, source.entries_[j].second);
        if (!entries_[i].second.is_zero()) result.push_back(std::move(entries_[i]));
        ++i;
        ++j;
      }
    }
    entries_.swap(result);
  }

 private:
  std::vector<Entry> entries_;
};

}  // namespace integral_native
