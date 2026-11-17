#pragma once

#include "compact_column.hpp"
#include "hybrid_column.hpp"

namespace integral_native {

/** Compact exact D*V storage with the scalar interface of the lazy engine.
 *
 * The transformation history retains actual rational column operations.
 * This adapter changes only explicit D*V: physical-row cancellation uses
 * the common-denominator kernel; a dummy-row cancellation applies the exact
 * factor queried from V. Small scalar queries avoid GMP construction. Full
 * entry conversion is diagnostic/export work, not a modular-pivot algorithm.
 */
class LazyCompactColumn {
 public:
  using Entry = HybridColumn::Entry;
  using ExportedEntry = HybridColumn::ExportedEntry;
  using SmallEntry = CompactColumn::SmallEntry;

  explicit LazyCompactColumn(std::vector<ExportedEntry> input)
      : column_(std::move(input)) {}

  static LazyCompactColumn from_integers(std::vector<IntegerColumn::IntegerEntry> input) {
    return LazyCompactColumn(CompactColumn::from_integers(std::move(input)));
  }

  static LazyCompactColumn from_sorted_small(std::vector<SmallEntry> entries,
                                             std::int64_t denominator) {
    return LazyCompactColumn(CompactColumn::from_sorted_small(std::move(entries), denominator));
  }

  std::size_t size() const noexcept { return column_.size(); }
  bool empty() const noexcept { return column_.empty(); }
  bool is_compact() const noexcept { return column_.is_compact(); }
  const std::vector<SmallEntry>& small_entries() const { return column_.small_entries(); }
  std::int64_t small_denominator() const { return column_.small_denominator(); }

  HybridRational coefficient_hybrid(int row) const {
    std::int64_t numerator, denominator;
    if (column_.small_coefficient(row, numerator, denominator))
      return HybridRational(numerator, denominator);
    return HybridRational(column_.coefficient(row));
  }
  mpq_class coefficient(int row) const { return column_.coefficient(row); }
  std::vector<ExportedEntry> entries() const { return column_.entries(); }

  /** Independent exact entry snapshot for diagnostic and export callers. */
  std::vector<Entry> hybrid_entries() const {
    std::vector<Entry> result;
    result.reserve(size());
    for (const auto& [row, value] : column_.entries())
      result.emplace_back(row, HybridRational(value));
    return result;
  }

  int canonical_pivot() const { return column_.canonical_pivot(); }
  int pivot(const std::vector<int>& positions) const { return column_.pivot(positions); }
  int modular_pivot(const mpz_class& prime, bool debug) const {
    return column_.modular_pivot(prime, debug);
  }
  int modular_pivot_or(const mpz_class& prime, int fallback, bool debug) const {
    return column_.modular_pivot_or(prime, fallback, debug);
  }

  void cancel_at(const LazyCompactColumn& source, int row) {
    column_.cancel_at(source.column_, row);
  }
  void subtract_hybrid(const LazyCompactColumn& source, HybridRational scalar) {
    if (this == &source) throw std::logic_error("self cancellation is forbidden");
    if (scalar.is_zero()) return;
    column_.subtract(source.column_, scalar.exact());
  }
  void subtract(const LazyCompactColumn& source, const mpq_class& scalar) {
    column_.subtract(source.column_, scalar);
  }

 private:
  explicit LazyCompactColumn(CompactColumn column) : column_(std::move(column)) {}
  CompactColumn column_;
};

}  // namespace integral_native
