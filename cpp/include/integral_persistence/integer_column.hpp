#pragma once

#include <gmpxx.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace integral_native {

/** Exact sparse rational column represented by A/d with integral A and d>0.
 *
 * The representation is globally primitive: gcd(d, A[0], ...) = 1. Storage
 * normalization divides numerator and denominator together and never rescales
 * the mathematical column. Unit-diagonal augmented transformations therefore
 * retain their exact values. A single denominator replaces per-entry mpq
 * normalization, while native GMP integers retain unbounded exactness.
 */
class IntegerColumn {
 public:
  using Entry = std::pair<int, mpq_class>;
  using Entries = std::vector<Entry>;
  using IntegerEntry = std::pair<int, mpz_class>;

  IntegerColumn() = default;
  /** Adopt an integral sparse column without copying its GMP coefficients. */
  static IntegerColumn from_integers(std::vector<IntegerEntry> entries) {
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    int previous = -1;
    for (const auto& [row, value] : entries) {
      if (row <= previous || value == 0)
        throw std::invalid_argument("integer entries require distinct nonnegative rows and nonzero values");
      previous = row;
    }
    IntegerColumn result;
    result.numerators_ = std::move(entries);
    return result;
  }

  explicit IntegerColumn(Entries entries) {
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.first < b.first; });
    int previous = -1;
    for (auto& [row, value] : entries) {
      if (row < 0 || row == previous)
        throw std::invalid_argument("column row IDs must be nonnegative and distinct");
      previous = row;
      if (mpz_sgn(mpq_denref(value.get_mpq_t())) == 0)
        throw std::invalid_argument("a rational coefficient has denominator zero");
      value.canonicalize();
      if (!value) throw std::invalid_argument("column entries must be nonzero");
      mpz_lcm(denominator_.get_mpz_t(), denominator_.get_mpz_t(), mpq_denref(value.get_mpq_t()));
    }
    numerators_.reserve(entries.size());
    for (const auto& [row, value] : entries) {
      mpz_class numerator;
      mpz_divexact(numerator.get_mpz_t(), denominator_.get_mpz_t(), mpq_denref(value.get_mpq_t()));
      mpz_mul(numerator.get_mpz_t(), numerator.get_mpz_t(), mpq_numref(value.get_mpq_t()));
      numerators_.emplace_back(row, std::move(numerator));
    }
    // Reduced entry fractions and their denominator lcm imply this already,
    // but keep one normalization boundary for all constructors.
    normalize();
  }

  std::size_t size() const noexcept { return numerators_.size(); }
  bool empty() const noexcept { return numerators_.empty(); }

  mpq_class coefficient(int row) const {
    const auto* value = find_numerator(row);
    if (!value) return 0;
    mpq_class answer(*value, denominator_);
    answer.canonicalize();
    return answer;
  }

  int canonical_pivot() const {
    if (empty()) throw std::logic_error("augmented column vanished");
    return numerators_.back().first;
  }

  /** Positions list rows from highest priority (zero) to lowest. */
  int pivot(const std::vector<int>& positions) const {
    int best = canonical_pivot();
    for (const auto& [row, value] : numerators_) {
      (void)value;
      if (positions[row] < positions[best]) best = row;
    }
    return best;
  }

  int modular_pivot(const mpz_class& prime, bool debug) const {
    if (prime < 2) throw std::invalid_argument("a modular pivot requires p >= 2");
    if (debug && mpz_divisible_p(denominator_.get_mpz_t(), prime.get_mpz_t()))
      throw std::logic_error("unsafe denominator in the current column");
    if (mpz_fits_ulong_p(prime.get_mpz_t())) {
      const auto small = prime.get_ui();
      for (auto it = numerators_.rbegin(); it != numerators_.rend(); ++it)
        if (!mpz_divisible_ui_p(it->second.get_mpz_t(), small)) return it->first;
    } else {
      for (auto it = numerators_.rbegin(); it != numerators_.rend(); ++it)
        if (!mpz_divisible_p(it->second.get_mpz_t(), prime.get_mpz_t())) return it->first;
    }
    throw std::logic_error("augmented column vanished modulo prime");
  }

  /** Exact target <- target - scalar*source, preserving source and all values. */
  void subtract(const IntegerColumn& source, const mpq_class& scalar) {
    if (!scalar || source.empty()) return;
    if (this == &source) {
      const IntegerColumn copied(source);
      subtract(copied, scalar);
      return;
    }
    if (denominator_ == 1 && source.denominator_ == 1 && scalar.get_den() == 1) {
      merge(source, denominator_, scalar.get_num());
      return;
    }
    // scalar = u/v. First cancel gcd(u,e) in scalar*(B/e), then use
    // lcm(d,v*e) to avoid unnecessarily multiplying denominators.
    mpz_class u = scalar.get_num();
    mpz_class source_denominator = source.denominator_;
    mpz_class common;
    mpz_gcd(common.get_mpz_t(), u.get_mpz_t(), source_denominator.get_mpz_t());
    if (common != 1) {
      mpz_divexact(u.get_mpz_t(), u.get_mpz_t(), common.get_mpz_t());
      mpz_divexact(source_denominator.get_mpz_t(), source_denominator.get_mpz_t(), common.get_mpz_t());
    }
    mpz_mul(source_denominator.get_mpz_t(), source_denominator.get_mpz_t(), mpq_denref(scalar.get_mpq_t()));
    mpz_gcd(common.get_mpz_t(), denominator_.get_mpz_t(), source_denominator.get_mpz_t());
    mpz_class target_scale, source_scale;
    mpz_divexact(target_scale.get_mpz_t(), source_denominator.get_mpz_t(), common.get_mpz_t());
    mpz_divexact(source_scale.get_mpz_t(), denominator_.get_mpz_t(), common.get_mpz_t());
    source_scale *= u;
    mpz_class new_denominator = denominator_ * target_scale;
    merge(source, target_scale, source_scale);
    denominator_.swap(new_denominator);
    normalize();
  }

  /** Pivot-aware exact cancellation with no per-entry rational arithmetic.
   *
   * For T=A/d, S=B/e, and cancellation row r, the new target is
   * (B[r]*A - A[r]*B)/(B[r]*d); e cancels algebraically. This optional
   * interface permits a reduction driver to avoid constructing the scalar.
   */
  void cancel_at(const IntegerColumn& source, int row) {
    if (this == &source) throw std::logic_error("self cancellation is forbidden");
    const auto* target_value = find_numerator(row);
    const auto* source_value = source.find_numerator(row);
    if (!target_value || !source_value)
      throw std::logic_error("cancellation row must be nonzero in both columns");
    mpz_class target_scale = *source_value;
    mpz_class source_scale = *target_value;
    if (target_scale < 0) {
      target_scale = -target_scale;
      source_scale = -source_scale;
    }
    // Cancel a scalar common factor before touching every sparse entry.
    // Dividing both multipliers and the new denominator by this same gcd
    // leaves the exact column unchanged and avoids temporary coefficient growth.
    if (target_scale != 1 && source_scale != 1 && source_scale != -1) {
      mpz_class common;
      mpz_gcd(common.get_mpz_t(), target_scale.get_mpz_t(), source_scale.get_mpz_t());
      if (common != 1) {
        mpz_divexact(target_scale.get_mpz_t(), target_scale.get_mpz_t(), common.get_mpz_t());
        mpz_divexact(source_scale.get_mpz_t(), source_scale.get_mpz_t(), common.get_mpz_t());
      }
    }
    mpz_class new_denominator = denominator_ * target_scale;
    merge(source, target_scale, source_scale);
    denominator_.swap(new_denominator);
    normalize();
  }

  Entries entries() const {
    Entries result;
    result.reserve(numerators_.size());
    for (const auto& [row, numerator] : numerators_) {
      mpq_class value(numerator, denominator_);
      value.canonicalize();
      result.emplace_back(row, std::move(value));
    }
    return result;
  }

  const mpz_class& denominator() const noexcept { return denominator_; }
  const std::vector<IntegerEntry>& integer_entries() const noexcept { return numerators_; }

 private:
  std::vector<IntegerEntry> numerators_;
  mpz_class denominator_ = 1;

  const mpz_class* find_numerator(int row) const {
    const auto it = std::lower_bound(numerators_.begin(), numerators_.end(), row,
        [](const IntegerEntry& entry, int id) { return entry.first < id; });
    return it != numerators_.end() && it->first == row ? &it->second : nullptr;
  }

  /** Merge target_scale*A - source_scale*B; denominator set by the caller. */
  void merge(const IntegerColumn& source, const mpz_class& target_scale, const mpz_class& source_scale) {
    std::vector<IntegerEntry> result;
    result.reserve(size() + source.size());
    std::size_t i = 0, j = 0;
    const bool target_unit = target_scale == 1;
    const int source_unit = source_scale == 1 ? 1 : (source_scale == -1 ? -1 : 0);
    auto append_source = [&](const IntegerEntry& entry) {
      mpz_class value;
      if (source_unit == 1) value = -entry.second;
      else if (source_unit == -1) value = entry.second;
      else { value = source_scale * entry.second; value = -value; }
      result.emplace_back(entry.first, std::move(value));
    };
    while (i < size() && j < source.size()) {
      if (numerators_[i].first < source.numerators_[j].first) {
        if (!target_unit) numerators_[i].second *= target_scale;
        result.push_back(std::move(numerators_[i++]));
      } else if (numerators_[i].first > source.numerators_[j].first) {
        append_source(source.numerators_[j++]);
      } else {
        auto& value = numerators_[i].second;
        if (!target_unit) value *= target_scale;
        if (source_unit == 1) value -= source.numerators_[j].second;
        else if (source_unit == -1) value += source.numerators_[j].second;
        else mpz_submul(value.get_mpz_t(), source_scale.get_mpz_t(), source.numerators_[j].second.get_mpz_t());
        if (value != 0) result.push_back(std::move(numerators_[i]));
        ++i;
        ++j;
      }
    }
    while (i < size()) {
      if (!target_unit) numerators_[i].second *= target_scale;
      result.push_back(std::move(numerators_[i++]));
    }
    while (j < source.size()) append_source(source.numerators_[j++]);
    numerators_.swap(result);
  }

  void normalize() {
    if (numerators_.empty()) { denominator_ = 1; return; }
    if (denominator_ == 1) return;
    mpz_class common = denominator_;
    for (const auto& [row, numerator] : numerators_) {
      (void)row;
      mpz_gcd(common.get_mpz_t(), common.get_mpz_t(), numerator.get_mpz_t());
      if (common == 1) return;
    }
    mpz_divexact(denominator_.get_mpz_t(), denominator_.get_mpz_t(), common.get_mpz_t());
    for (auto& [row, numerator] : numerators_) {
      (void)row;
      mpz_divexact(numerator.get_mpz_t(), numerator.get_mpz_t(), common.get_mpz_t());
    }
  }
};

}  // namespace integral_native
