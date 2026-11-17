#pragma once

#include "lazy_compact_column.hpp"
#include "compact_cancel_factor.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace integral_native {

/** Reusable exact working column with dense numerators and one denominator.
 *
 * The working value is A/d, never a projectively rescaled column. If the
 * pivot-aware common-denominator update has unit target multiplier, only
 * source support is visited. Checked arithmetic restores the already updated
 * prefix on overflow. Large sources
 * and unrepresentable updates use the exact compact/GMP kernel from the
 * untouched current state. Final sparse export normalizes A and d together.
 * Nonunit scales use reusable signed-128 scratch, normalizing only when
 * needed to retain word-sized data.
 *
 * Every cancellation returns its original rational factor for V's operation
 * log. Sources remain immutable; no assumption about boundary dimensions,
 * input signs, matrix rank, or small coefficient sizes is used.
 */
class DenseCompactWorkingColumn {
 public:
  using Wide = __int128_t;
  using UnsignedWide = __uint128_t;

  DenseCompactWorkingColumn(int row_count, int row_offset)
      : row_offset_(checked_offset(row_count, row_offset)),
        values_(static_cast<std::size_t>(row_count), 0), support_(values_.size()),
        scratch_(values_.size(), 0) {}
  DenseCompactWorkingColumn(const DenseCompactWorkingColumn&) = delete;
  DenseCompactWorkingColumn& operator=(const DenseCompactWorkingColumn&) = delete;
  DenseCompactWorkingColumn(DenseCompactWorkingColumn&&) noexcept = default;
  DenseCompactWorkingColumn& operator=(DenseCompactWorkingColumn&&) noexcept = default;

  std::size_t size() const noexcept { return fallback_ ? fallback_->size() : nonzeros_; }
  bool empty() const noexcept { return size() == 0; }

  void load(const LazyCompactColumn& source) {
    clear();
    if (!source.is_compact()) {
      // Validate physical labels even when the representation is already GMP.
      for (const auto& entry : source.entries()) (void)local_index(entry.first);
      fallback_ = std::make_unique<LazyCompactColumn>(source);
      return;
    }
    denominator_ = source.small_denominator();
    for (const auto& [row, numerator] : source.small_entries()) {
      const auto index = local_index(row);
      values_[index] = numerator;
      support_.insert(index);
      ++nonzeros_;
    }
  }

  void clear() {
    fallback_.reset();
    support_.consume([&](std::size_t index) { values_[index] = 0; });
    nonzeros_ = 0;
    denominator_ = 1;
  }

  int canonical_pivot() const {
    if (empty()) return -1;
    return fallback_ ? fallback_->canonical_pivot()
                     : row_offset_ + static_cast<int>(support_.highest());
  }

  HybridRational coefficient_hybrid(int row) const {
    const auto index = local_index(row);
    if (fallback_) return fallback_->coefficient_hybrid(row);
    return HybridRational(values_[index], denominator_);
  }

  /** Cancel a nonzero physical row and return the exact pre-update factor. */
  HybridRational cancel_at(const LazyCompactColumn& source, int row) {
    if (fallback_ || !source.is_compact()) {
      const auto target = coefficient_hybrid(row);
      const auto pivot = source.coefficient_hybrid(row);
      if (target.is_zero() || pivot.is_zero())
        throw std::domain_error("cancellation row must be nonzero in both columns");
      auto factor = HybridRational::quotient(target, pivot);

      fallback_cancel(source, row);
      return factor;
    }

    const auto& entries = source.small_entries();
    if (entries.empty()) throw std::domain_error("source cancellation coefficient is zero");
    // A canonical source's pivot is its final entry. Other callers retain
    // the general exact row lookup.
    const auto found = entries.back().first == row ? entries.end() - 1 :
        std::lower_bound(entries.begin(), entries.end(), row,
            [](const auto& entry, int key) { return entry.first < key; });
    if (found == entries.end() || found->first != row)
      throw std::domain_error("source cancellation coefficient is zero");
    const auto target_pivot = values_[local_index(row)];
    auto cancellation = compact_cancellation(target_pivot, denominator_, found->second,
                                             source.small_denominator());

    if (cancellation.target_scale != 1) {
      if (dense_general_cancel(source, cancellation))
        return std::move(cancellation.factor);
      fallback_cancel(source, row);
      return std::move(cancellation.factor);
    }

    // Validate the entire sorted row range before modifying any coefficient.
    (void)local_index(entries.front().first);
    (void)local_index(entries.back().first);

    const Wide scale = cancellation.source_scale;
    for (std::size_t processed = 0; processed < entries.size(); ++processed) {
      const auto& [source_row, numerator] = entries[processed];
      const auto index = static_cast<std::size_t>(source_row - row_offset_);
      std::int64_t value;
      bool fits;
      if (scale == 1) {
        fits = !__builtin_sub_overflow(values_[index], numerator, &value);
      } else if (scale == -1) {
        fits = !__builtin_add_overflow(values_[index], numerator, &value);
      } else {
        // Both multiplicands have magnitude at most 2**63, including an
        // INT64_MIN numerator or positive 2**63 scale. The product is at
        // most 2**126 and subtracting it from int64 still fits signed Wide.
        const Wide product = scale * static_cast<Wide>(numerator);
        const Wide difference = static_cast<Wide>(values_[index]) - product;
        fits = difference >= std::numeric_limits<std::int64_t>::min() &&
               difference <= std::numeric_limits<std::int64_t>::max();
        if (fits) value = static_cast<std::int64_t>(difference);
      }
      if (!fits) {
        // Sources have distinct immutable rows. Inverting exactly the already
        // applied prefix restores both coefficients and support. Products fit
        // signed 128 bits; each restored scalar equals its original int64.
        for (std::size_t previous = 0; previous < processed; ++previous) {
          const auto& [previous_row, previous_numerator] = entries[previous];
          const auto previous_index = static_cast<std::size_t>(previous_row - row_offset_);
          const Wide product = scale * static_cast<Wide>(previous_numerator);
          const Wide restored = static_cast<Wide>(values_[previous_index]) + product;
          if (restored < std::numeric_limits<std::int64_t>::min() ||
              restored > std::numeric_limits<std::int64_t>::max())
            throw std::logic_error("exact compact rollback invariant failed");
          set_value(previous_index, static_cast<std::int64_t>(restored));
        }

        fallback_cancel(source, row);
        return std::move(cancellation.factor);
      }
      set_value(index, value);
    }
    if (!nonzeros_) denominator_ = 1;

    return std::move(cancellation.factor);
  }

  /** Transfer a sorted, normalized exact column and reset reusable storage. */
  LazyCompactColumn release_column() {
    if (fallback_) {
      auto result = std::move(*fallback_);
      fallback_.reset();
      return result;
    }
    std::vector<LazyCompactColumn::SmallEntry> result;
    result.reserve(nonzeros_);
    support_.consume([&](std::size_t index) {
      result.emplace_back(row_offset_ + static_cast<int>(index), values_[index]);
      values_[index] = 0;
    });
    const auto denominator = denominator_;
    denominator_ = 1;
    nonzeros_ = 0;
    return LazyCompactColumn::from_sorted_small(std::move(result), denominator);
  }

 private:
  /** Apply one exact machine-word change, symmetrically usable for rollback. */
  void set_value(std::size_t index, std::int64_t value) {
    if (values_[index] == 0 && value != 0) {
      support_.insert(index);
      ++nonzeros_;
    } else if (values_[index] != 0 && value == 0) {
      support_.erase(index);
      --nonzeros_;
    }
    values_[index] = value;
  }

  /** Form and validate a*A-b*B, normalizing only when needed for word bounds.
   *
   * Scratch rows are reset after every attempt. Seeding visits only nonzero
   * target entries; each immutable source has distinct rows. Consequently a
   * zero scratch slot at its source visit identifies a previously untouched
   * row, even though the resulting value may become zero. No row stamps or
   * per-update dense clearing are necessary.
   */
  bool dense_general_cancel(const LazyCompactColumn& source,
                            const CompactCancellation& cancellation) {
    const auto& entries = source.small_entries();
    (void)local_index(entries.front().first);
    (void)local_index(entries.back().first);
    touched_.clear();
    touched_.reserve(nonzeros_ + entries.size());
    const Wide a = cancellation.target_scale, b = cancellation.source_scale;
    const Wide denominator = a * static_cast<Wide>(denominator_);
    support_.visit([&](std::size_t index) {
      scratch_[index] = a * static_cast<Wide>(values_[index]);
      touched_.push_back(index);
    });
    auto discard = [&]() {
      for (const auto index : touched_) scratch_[index] = 0;
      touched_.clear();
      return false;
    };
    for (const auto& [row, value] : entries) {
      const auto index = static_cast<std::size_t>(row - row_offset_);
      if (scratch_[index] == 0) touched_.push_back(index);
      // Every product fits signed Wide because each operand has magnitude
      // at most 2**63. Keep a checked subtraction at the representation
      // boundary, including the most negative signed-128 numerator.
      const Wide product = b * static_cast<Wide>(value);
      Wide difference;
      if (__builtin_sub_overflow(scratch_[index], product, &difference)) return discard();
      scratch_[index] = difference;
    }

    Wide reduced_denominator = denominator;
    bool raw_fits = denominator <= std::numeric_limits<std::int64_t>::max();
    if (raw_fits) {
      for (const auto index : touched_) {
        if (scratch_[index] < std::numeric_limits<std::int64_t>::min() ||
            scratch_[index] > std::numeric_limits<std::int64_t>::max()) {
          raw_fits = false;
          break;
        }
      }
    }
    // As in unit updates, a working column need not be primitive. Preserve
    // its exact A/d directly when all words fit; normalize only to recover
    // bounds or at the final sparse export, before any modular pivot query.
    if (!raw_fits) {
      UnsignedWide common = static_cast<UnsignedWide>(denominator);
      for (const auto index : touched_) {
        if (common == 1) break;
        const Wide value = scratch_[index];
        UnsignedWide numerator = value < 0 ? UnsignedWide(-(value + 1)) + 1 : UnsignedWide(value);
        while (numerator) {
          const UnsignedWide remainder = common % numerator;
          common = numerator;
          numerator = remainder;
        }
      }
      // common divides a positive denominator below 2**127, hence this cast
      // and every exact division are safe even for a numerator INT128_MIN.
      const Wide divisor = static_cast<Wide>(common);
      reduced_denominator = denominator / divisor;
      if (reduced_denominator > std::numeric_limits<std::int64_t>::max()) return discard();
      for (const auto index : touched_) {
        if (divisor != 1) scratch_[index] /= divisor;
        if (scratch_[index] < std::numeric_limits<std::int64_t>::min() ||
            scratch_[index] > std::numeric_limits<std::int64_t>::max()) return discard();
      }
    }
    // All validated coefficients now fit. Only this final loop mutates the
    // actual column; failures above leave it intact for the exact GMP path.
    for (const auto index : touched_) {
      set_value(index, static_cast<std::int64_t>(scratch_[index]));
      scratch_[index] = 0;
    }
    denominator_ = nonzeros_ ? static_cast<std::int64_t>(reduced_denominator) : 1;
    touched_.clear();
    return true;
  }

  /** Hierarchical support bitmap; upper bits identify nonzero lower words. */
  class Support {
   public:
    explicit Support(std::size_t rows) {
      if (!rows) return;
      std::size_t words = (rows + 63) / 64;
      while (true) {
        levels_.emplace_back(words, std::uint64_t(0));
        if (words == 1) break;
        words = (words + 63) / 64;
      }
    }
    void insert(std::size_t index) {
      for (auto& level : levels_) {
        auto& word = level[index / 64];
        const bool was_empty = word == 0;
        word |= std::uint64_t(1) << (index % 64);
        if (!was_empty) break;
        index /= 64;
      }
    }
    void erase(std::size_t index) {
      for (auto& level : levels_) {
        auto& word = level[index / 64];
        word &= ~(std::uint64_t(1) << (index % 64));
        if (word) break;
        index /= 64;
      }
    }
    std::size_t highest() const {
      std::size_t index = 0;
      for (std::size_t depth = levels_.size(); depth-- > 0;) {
        const auto word = levels_[depth][index];
        index = index * 64 + 63 - static_cast<unsigned>(__builtin_clzll(word));
      }
      return index;
    }
    template <class Visitor>
    void consume(Visitor&& visit) {
      if (!levels_.empty()) consume_word(levels_.size() - 1, 0, visit);
    }
    template <class Visitor>
    void visit(Visitor&& visitor) const {
      if (!levels_.empty()) visit_word(levels_.size() - 1, 0, visitor);
    }

   private:
    std::vector<std::vector<std::uint64_t>> levels_;
    template <class Visitor>
    void visit_word(std::size_t depth, std::size_t index, Visitor& visit) const {
      auto bits = levels_[depth][index];
      while (bits) {
        const auto child = index * 64 + static_cast<unsigned>(__builtin_ctzll(bits));
        if (depth) visit_word(depth - 1, child, visit);
        else visit(child);
        bits &= bits - 1;
      }
    }
    template <class Visitor>
    void consume_word(std::size_t depth, std::size_t index, Visitor& visit) {
      auto bits = levels_[depth][index];
      while (bits) {
        const auto child = index * 64 + static_cast<unsigned>(__builtin_ctzll(bits));
        if (depth) consume_word(depth - 1, child, visit);
        else visit(child);
        bits &= bits - 1;
      }
      levels_[depth][index] = 0;
    }
  };

  int row_offset_;
  std::vector<std::int64_t> values_;
  Support support_;
  std::size_t nonzeros_ = 0;
  std::int64_t denominator_ = 1;

  std::unique_ptr<LazyCompactColumn> fallback_;

  std::vector<Wide> scratch_;
  std::vector<std::size_t> touched_;

  static int checked_offset(int rows, int offset) {
    if (rows < 0 || offset < 0 || rows > std::numeric_limits<int>::max() - offset)
      throw std::invalid_argument("working column row range exceeds signed indices");
    return offset;
  }
  std::size_t local_index(int row) const {
    if (row < row_offset_ || static_cast<std::size_t>(row - row_offset_) >= values_.size())
      throw std::out_of_range("working column row lies outside the explicit boundary block");
    return static_cast<std::size_t>(row - row_offset_);
  }
  void fallback_cancel(const LazyCompactColumn& source, int row) {
    if (fallback_) {
      fallback_->cancel_at(source, row);
      if (fallback_->is_compact()) {
        auto small = std::move(*fallback_);
        fallback_.reset();
        load(small);
      }
      return;
    }
    auto column = release_column();
    column.cancel_at(source, row);
    if (column.is_compact()) load(column);
    else fallback_ = std::make_unique<LazyCompactColumn>(std::move(column));
  }

};

}  // namespace integral_native
