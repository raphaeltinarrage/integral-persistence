#pragma once

#include "engine_api.hpp"
#include "hybrid_column.hpp"
#include "lazy_compact_column.hpp"
#include "compact_working_column.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace integral_native {

/** Algorithms 1–5 with explicit D*V and an exact persistent history for V.
 *
 * Every operation target <- target - scalar*source is retained in order with
 * its exact rational scalar. Transformation rows needed by priority updates
 * are replayed on demand and subsequently updated. The history is never
 * replaced by a fresh identity after canonical reduction. Full augmented
 * columns can be reconstructed exactly by internal invariant checks.
 *
 * The canonical/exhaustive shortcut preserves the exact augmented matrix.
 * Arbitrary priorities inspect actual lazily evaluated transformation
 * coefficients, not a truncated state.
 */
class LazyCompactDenseEngine final : public NativeEngine {
 public:
  /** Consume a validated integer matrix without retaining an original copy.
   *
   * Each sparse integer column is transferred into compact or GMP storage in
   * turn. Branches share only the current reduction and persistent history.
   */
  LazyCompactDenseEngine(int rows, KernelColumns&& input, KernelPoll poll = {})
      : original_rows_(rows), column_count_(checked_shape(rows, input.size())),
        poll_(std::move(poll)) {
    for (const auto& column : input) {
      std::unordered_set<int> seen;
      for (const auto& [row, value] : column)
        if (row < 0 || row >= rows || value == 0 || !seen.insert(row).second)
          throw std::invalid_argument("input columns require distinct valid rows and nonzero coefficients");
    }
    priority_.resize(nrows());
    std::iota(priority_.rbegin(), priority_.rend(), 0);
    position_.resize(nrows());
    install_positions();
    owner_.assign(nrows(), -1);
    first_source_use_.assign(ncols(), no_operation);
    incidence_.resize(ncols());
    columns_.reserve(ncols());
    for (auto& column : input) {
      poll_now();
      for (auto& entry : column) entry.first += ncols();
      columns_.push_back(std::make_shared<LazyCompactColumn>(
          LazyCompactColumn::from_integers(std::move(column))));
    }
    KernelColumns{}.swap(input);
  }

  int nrows() const override { return original_rows_ + column_count_; }
  int ncols() const override { return column_count_; }
  void set_poll(KernelPoll poll) override { poll_ = std::move(poll); }
  const std::vector<int>& priority() const override { return priority_; }
  const std::vector<int>& owner() const override { return owner_; }
  std::shared_ptr<NativeEngine> fork() const override {
    return std::make_shared<LazyCompactDenseEngine>(*this);
  }

  std::vector<int> pivots() const override {
    require_reduced();
    std::vector<int> result(ncols(), -1);
    for (int row = 0; row < nrows(); ++row)
      if (owner_[row] != -1) result[owner_[row]] = row;
    return result;
  }

  void reduce(bool debug = false) override {
    seed_identity_rows_ = false;
    reduce_impl(debug, false);
  }

  void exhaustive(bool debug = false) override {
    require_reduced();
    seed_identity_rows_ = false;
    if (!is_canonical()) throw std::invalid_argument("exhaustive preparation requires canonical priority");
    if (debug) assert_valid();
    auto order = pivots();
    bool identity_births = true;
    for (int j = 0; j < ncols(); ++j)
      if (order[j] < ncols() && first_source_use_[j] != no_operation) identity_births = false;
    std::vector<bool> used(nrows(), false);
    for (int row : order) used[row] = true;
    for (int row : priority_) if (!used[row]) order.push_back(row);
    priority_.swap(order);
    install_positions();
    // Other dummy rows lie below all original rows; birth rows are identity
    // rows and never conflict. If historical generic updates destroyed that
    // condition, use full exact lazy pivot queries instead.
    reduce_impl(debug, identity_births);
  }

  void transpose(int index, bool debug = false) override {
    require_reduced();
    seed_identity_rows_ = true;
    if (index < 0 || index >= nrows() - 1) throw std::out_of_range("transposition outside priority");
    transpose_unchecked(index);
    if (debug) assert_valid();
  }

  void correct_priority(const std::vector<int>& target, bool debug = false) override {
    require_reduced();
    seed_identity_rows_ = true;
    validate_priority(target);
    for (int i = 0; i < nrows(); ++i) move_row(target[i], i, debug);
    if (debug) assert_valid();
  }

  void correct_prime(const std::string& text, bool debug = false) override {
    require_exhaustive();
    seed_identity_rows_ = true;
    const auto prime = parse_integer(text);
    if (prime < 2) throw std::invalid_argument("characteristic must exceed one");
    for (int j = 0; j < ncols(); ++j) {
      poll_now();
      const int row = modular_pivot(j, prime, debug);
      move_row(row, j, debug);
      if (owner_[row] != j) throw std::logic_error("correction lost current pivot owner");
    }
  }

  GroupResult correct_groups(const std::vector<std::string>& texts, bool debug = false) override {
    require_exhaustive();
    seed_identity_rows_ = true;
    struct Branch { std::vector<std::size_t> primes; std::shared_ptr<LazyCompactDenseEngine> state; };
    std::vector<mpz_class> primes;
    for (const auto& text : texts) {
      primes.push_back(parse_integer(text));
      if (primes.back() < 2) throw std::invalid_argument("characteristic must exceed one");
    }
    GroupResult result;
    if (primes.empty()) return result;
    std::vector<std::size_t> indices(primes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<Branch> groups;
    groups.push_back({std::move(indices), std::static_pointer_cast<LazyCompactDenseEngine>(shared_from_this())});
    for (int j = 0; j < ncols(); ++j) {
      std::vector<Branch> children;
      poll_now();
      for (auto& group : groups) {
        std::vector<std::pair<int, std::vector<std::size_t>>> buckets;
        for (auto p : group.primes) {
          const int row = group.state->modular_pivot(j, primes[p], debug);
          auto found = std::find_if(buckets.begin(), buckets.end(),
                                   [row](const auto& bucket) { return bucket.first == row; });
          if (found == buckets.end()) buckets.push_back({row, {p}});
          else found->second.push_back(p);
        }
        for (auto& [row, members] : buckets) {
          auto child = buckets.size() == 1 ? group.state : std::make_shared<LazyCompactDenseEngine>(*group.state);
          child->move_row(row, j, debug);
          if (child->owner_[row] != j) throw std::logic_error("group correction lost pivot owner");
          children.push_back({std::move(members), std::move(child)});
        }
      }
      groups.swap(children);
    }
    for (auto& group : groups) {
      std::vector<std::string> members;
      for (auto p : group.primes) members.push_back(texts[p]);
      result.groups.push_back({std::move(members), std::move(group.state)});
    }
    return result;
  }

  std::vector<std::string> minors() const override {
    require_reduced();
    if (!is_canonical()) throw std::invalid_argument("minor discovery requires canonical state");
    const auto low = pivots();
    mpq_class delta = 1;
    std::vector<std::string> result;
    result.reserve(ncols());
    for (int j = 0; j < ncols(); ++j) {
      poll_now();
      delta *= coefficient(low[j], j).exact();
      if (delta == 0 || delta.get_den() != 1)
        throw std::logic_error("cumulative pivot product is not a nonzero integer");
      result.push_back(delta.get_num().get_str());
    }
    return result;
  }

  void assert_valid() const override {
    require_reduced();
    validate_priority(priority_);
    for (int row = 0; row < nrows(); ++row)
      if (position_[priority_[row]] != row) throw std::logic_error("invalid inverse priority");
    const auto transform = materialize_transform();
    std::vector<int> expected_owner(nrows(), -1);
    for (int j = 0; j < ncols(); ++j) {
      if (transform[j].coefficient(j) != 1) throw std::logic_error("transformation diagonal is not one");
      for (const auto& [row, value] : transform[j].hybrid_entries())
        if (row > j || row < 0 || value.is_zero())
          throw std::logic_error("invalid upper unitriangular transform");
      int low = transform[j].pivot(position_);
      if (columns_[j]->size()) {
        const int lower = columns_[j]->pivot(position_);
        if (position_[lower] < position_[low]) low = lower;
      }
      if (expected_owner[low] != -1) throw std::logic_error("duplicate pivot");
      expected_owner[low] = j;
      for (const auto& [row, values] : row_cache_) {
        const auto cached = values->coefficient(j);
        if (cached.exact() != transform[j].coefficient(row))
          throw std::logic_error("cached transformation row disagrees with exact log replay");
        const auto& incident = incidence_[j];
        const auto count = incident ? std::count(incident->begin(), incident->end(), row) : 0;
        if (count != (cached.is_zero() ? 0 : 1))
          throw std::logic_error("cached-row incidence disagrees with nonzero coefficients");
      }
      if (incidence_[j]) {
        for (int row : *incidence_[j]) {
          const auto found = row_cache_.find(row);
          if (found == row_cache_.end() || found->second->coefficient(j).is_zero())
            throw std::logic_error("stale cached-row incidence");
        }
      }
    }
    if (expected_owner != owner_) throw std::logic_error("owner array differs from pivots");
  }

  /** Independent invariant hook: verify R=A V against externally held input.
   * No original coefficient copy is stored by the engine itself.
   */
  void assert_product(const KernelColumns& original) const override {
    require_reduced();
    if (original.size() != static_cast<std::size_t>(ncols()))
      throw std::invalid_argument("original column count differs from reduction");
    const auto transform = materialize_transform();
    for (int j = 0; j < ncols(); ++j) {
      std::map<int, mpq_class> expected;
      for (const auto& [source, scalar] : transform[j].hybrid_entries())
        for (const auto& [row, value] : original[source])
          expected[row + ncols()] += scalar.exact() * value;
      for (auto it = expected.begin(); it != expected.end();)
        if (it->second == 0) it = expected.erase(it); else ++it;
      if (expected.size() != columns_[j]->size())
        throw std::logic_error("augmented identity R = A V failed");
      for (const auto& [row, value] : columns_[j]->hybrid_entries()) {
        const auto found = expected.find(row);
        if (found == expected.end() || value.exact() != found->second)
          throw std::logic_error("augmented identity R = A V failed");
      }
    }
  }

 private:
  /** Sparse current transformation row, sorted by column index. */
  struct Row {
    using Entry = std::pair<int, HybridRational>;
    std::vector<Entry> entries;
    explicit Row(int row) { entries.emplace_back(row, HybridRational(mpq_class(1))); }
    HybridRational coefficient(int column) const {
      const auto found = std::lower_bound(entries.begin(), entries.end(), column,
          [](const Entry& entry, int index) { return entry.first < index; });
      return found != entries.end() && found->first == column ? found->second : HybridRational();
    }
    // Return +1 if the target entered the support, -1 if it left, and zero
    // otherwise. Source and target differ, so insertion cannot invalidate a
    // source value before its arithmetic has completed.
    int update(int target, int source, const HybridRational& factor) {
      const auto source_it = std::lower_bound(entries.begin(), entries.end(), source,
          [](const Entry& entry, int index) { return entry.first < index; });
      if (source_it == entries.end() || source_it->first != source) return 0;
      const auto target_it = std::lower_bound(entries.begin(), entries.end(), target,
          [](const Entry& entry, int index) { return entry.first < index; });
      if (target_it != entries.end() && target_it->first == target) {
        target_it->second.subtract_product(factor, source_it->second);
        if (target_it->second.is_zero()) { entries.erase(target_it); return -1; }
        return 0;
      }
      HybridRational value;
      value.subtract_product(factor, source_it->second);
      if (value.is_zero()) return 0;
      entries.emplace(target_it, target, std::move(value));
      return 1;
    }
  };
  static constexpr std::size_t chunk_capacity = 1024;
  static constexpr std::size_t no_operation = std::numeric_limits<std::size_t>::max();
  struct Operation { int target, source; HybridRational factor; };
  struct LogChunk {
    std::shared_ptr<LogChunk> previous;
    std::vector<Operation> operations;
    std::size_t start;
    LogChunk(std::shared_ptr<LogChunk> parent, std::size_t offset)
        : previous(std::move(parent)), start(offset) { operations.reserve(chunk_capacity); }
    // Release a long unshared history iteratively rather than recursing once
    // per chunk when the last state goes away.
    ~LogChunk() {
      auto ancestor = std::move(previous);
      while (ancestor && ancestor.unique()) {
        auto next = std::move(ancestor->previous);
        ancestor.reset();
        ancestor = std::move(next);
      }
    }
  };

  int original_rows_, column_count_;
  KernelPoll poll_;
  mutable std::size_t poll_count_ = 0;
  std::vector<std::shared_ptr<LazyCompactColumn>> columns_;
  std::vector<int> priority_, position_, owner_;
  std::vector<std::size_t> first_source_use_;
  std::shared_ptr<LogChunk> tail_;
  std::size_t log_size_ = 0;
  mutable std::unordered_map<int, std::shared_ptr<Row>> row_cache_;
  // Column -> cached row IDs whose coefficient in this column is nonzero.
  // Each list is detached only when a child changes that column's incidence.
  mutable std::vector<std::shared_ptr<std::vector<int>>> incidence_;
  bool reduced_ = false;
  bool seed_identity_rows_ = false;

  static int checked_shape(int rows, std::size_t columns) {
    if (rows < 0 || columns > static_cast<std::size_t>(std::numeric_limits<int>::max() - rows))
      throw std::invalid_argument("augmented matrix exceeds signed index range");
    return static_cast<int>(columns);
  }
  void poll_now() const { if (poll_) poll_(); }
  void poll_periodically() const {
    if ((++poll_count_ & 1023U) == 0) poll_now();
  }
  void require_reduced() const {
    if (!reduced_) throw std::invalid_argument("state must first be reduced");
  }
  void require_exhaustive() const {
    const auto low = pivots();
    if (!std::equal(low.begin(), low.end(), priority_.begin()))
      throw std::invalid_argument("correction requires exhaustive pivot priority");
  }
  void validate_priority(const std::vector<int>& order) const {
    if (order.size() != static_cast<std::size_t>(nrows())) throw std::invalid_argument("invalid priority length");
    std::vector<bool> seen(nrows(), false);
    for (int row : order) {
      if (row < 0 || row >= nrows() || seen[row]) throw std::invalid_argument("priority is not a permutation");
      seen[row] = true;
    }
  }
  bool is_canonical() const {
    for (int i = 0; i < nrows(); ++i) if (priority_[i] != nrows() - 1 - i) return false;
    return true;
  }
  void install_positions() {
    for (int i = 0; i < nrows(); ++i) position_[priority_[i]] = i;
  }

  template <class Visitor>
  void visit_log(Visitor&& visit, std::size_t begin = 0) const {
    std::vector<const LogChunk*> chunks;
    for (auto* chunk = tail_.get(); chunk; chunk = chunk->previous.get()) {
      if (chunk->start + chunk->operations.size() <= begin) break;
      chunks.push_back(chunk);
    }
    for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
      const auto* chunk = *it;
      const auto offset = begin > chunk->start ? begin - chunk->start : 0;
      for (std::size_t i = offset; i < chunk->operations.size(); ++i) visit(chunk->operations[i]);
    }
  }
  std::vector<HybridColumn> materialize_transform() const {
    std::vector<HybridColumn> result;
    result.reserve(ncols());
    for (int j = 0; j < ncols(); ++j)
      result.emplace_back(std::vector<std::pair<int, mpq_class>>{{j, 1}});
    visit_log([&](const Operation& op) { result[op.target].subtract_hybrid(result[op.source], op.factor); });
    return result;
  }
  const Row& transformation_row(int row) const {
    const auto found = row_cache_.find(row);
    if (found != row_cache_.end()) return *found->second;
    auto values = std::make_shared<Row>(row);
    if (first_source_use_[row] != no_operation) {
      visit_log([&](const Operation& op) {
        poll_periodically();
        values->update(op.target, op.source, op.factor);
      }, first_source_use_[row]);
    }
    const auto inserted = row_cache_.emplace(row, std::move(values));
    for (const auto& [column, value] : inserted.first->second->entries) {
      (void)value;
      change_incidence(column, row, true);
    }
    return *inserted.first->second;
  }
  HybridRational coefficient(int row, int column) const {
    if (row >= ncols()) return columns_[column]->coefficient_hybrid(row);
    if (row > column) return HybridRational();
    if (row == column) return HybridRational(mpq_class(1));
    if (first_source_use_[row] == no_operation) return HybridRational();
    return transformation_row(row).coefficient(column);
  }

  void change_incidence(int column, int row, bool insert) const {
    auto& rows = incidence_[column];
    if (!rows) {
      if (!insert) throw std::logic_error("missing cached-row incidence");
      rows = std::make_shared<std::vector<int>>();
    } else if (!rows.unique()) {
      rows = std::make_shared<std::vector<int>>(*rows);
    }
    if (insert) {
      rows->push_back(row);
    } else {
      const auto found = std::find(rows->begin(), rows->end(), row);
      if (found == rows->end()) throw std::logic_error("missing cached-row incidence");
      *found = rows->back();
      rows->pop_back();
    }
  }

  int full_pivot(int column, bool identity_births) const {
    int low = column;
    if (columns_[column]->size()) {
      const int candidate = columns_[column]->pivot(position_);
      if (position_[candidate] < position_[low]) low = candidate;
    }
    if (identity_births) return low;
    for (int index = 0; index < position_[low]; ++index) {
      const int row = priority_[index];
      if (row < column && !coefficient(row, column).is_zero()) return row;
    }
    return low;
  }
  void reduce_impl(bool debug, bool identity_births) {
    std::fill(owner_.begin(), owner_.end(), -1);
    const bool canonical = is_canonical();
    if (canonical) { reduce_working(debug); return; }
    for (int j = 0; j < ncols(); ++j) {
      poll_now();
      int row = full_pivot(j, identity_births);
      while (owner_[row] != -1) {
        cancel(j, owner_[row], row);
        row = full_pivot(j, identity_births);
      }
      owner_[row] = j;
    }
    reduced_ = true;
    if (debug) assert_valid();
  }

  /** Canonical reduction with one reusable exact accumulator for D*V.
   *
   * Finished sources remain sparse. Each cancellation updates only source
   * support; its original rational scalar is appended to V's chronological
   * history. The mathematical column is never rescaled or discarded.
   */
  void reduce_working(bool debug) {
    DenseCompactWorkingColumn work(original_rows_, ncols());
    for (int j = 0; j < ncols(); ++j) {
      poll_now();
      int row = columns_[j]->size() ? columns_[j]->canonical_pivot() : j;
      if (owner_[row] != -1) {
        work.load(*columns_[j]);
        do {
          const int source = owner_[row];
          if (source >= j) throw std::logic_error("invalid earlier-column cancellation");
          const auto factor = work.cancel_at(*columns_[source], row);
          append_operation(j, source, factor);
          row = work.size() ? work.canonical_pivot() : j;
        } while (owner_[row] != -1);
        columns_[j] = std::make_shared<LazyCompactColumn>(work.release_column());
      }
      owner_[row] = j;
    }
    reduced_ = true;
    if (debug) assert_valid();
  }

  int modular_pivot(int j, const mpz_class& prime, bool debug) {
    if (debug) {
      for (const auto& [row, value] : columns_[j]->hybrid_entries()) {
        (void)row;
        if (value.denominator_divisible(prime)) throw std::logic_error("unsafe denominator in current column");
      }
      // Debug checks the hidden block too. A safe lower block alone does not
      // prove that the full current augmented column has p-unit denominators.
      const auto transform = materialize_transform();
      for (const auto& [row, value] : transform[j].hybrid_entries()) {
        (void)row;
        if (value.denominator_divisible(prime)) throw std::logic_error("unsafe hidden denominator in current column");
      }
    }
    const int answer = columns_[j]->modular_pivot_or(prime, j, debug);
    return answer;
  }

  void cancel(int target, int source, int row) {
    if (source >= target) throw std::logic_error("invalid earlier-column cancellation");

    const auto factor = HybridRational::quotient(coefficient(row, target), coefficient(row, source));
    if (factor.is_zero()) throw std::logic_error("zero cancellation numerator");
    if (columns_[source]->size()) {
      if (!columns_[target].unique()) {
        columns_[target] = std::make_shared<LazyCompactColumn>(*columns_[target]);
      }
      if (row >= ncols()) columns_[target]->cancel_at(*columns_[source], row);
      else columns_[target]->subtract_hybrid(*columns_[source], factor);
    }
    append_operation(target, source, factor);
  }

  /** Record the exact unscaled operation and update every materialized V row. */
  void append_operation(int target, int source, const HybridRational& factor) {
    // A never-used source still has an identity transformation row. During
    // corrections, seed that tiny row at its first use rather than replaying
    // its entire subsequent history when its dummy pivot is later promoted.
    // Canonical/exhaustive reduction deliberately leaves this optimization off.
    if (seed_identity_rows_ && first_source_use_[source] == no_operation)
      transformation_row(source);
    if (incidence_[source]) {
      for (int cached_row : *incidence_[source]) {
        auto& values = row_cache_.at(cached_row);
        if (!values.unique()) { values = std::make_shared<Row>(*values); }
        const int transition = values->update(target, source, factor);
        if (transition) change_incidence(target, cached_row, transition > 0);
        poll_periodically();
      }
    }
    if (first_source_use_[source] == no_operation) first_source_use_[source] = log_size_;
    if (!tail_ || !tail_.unique() || tail_->operations.size() == chunk_capacity)
      tail_ = std::make_shared<LogChunk>(tail_, log_size_);
    tail_->operations.push_back({target, source, factor});
    ++log_size_;
    poll_periodically();
  }

  void transpose_unchecked(int index) {
    const int u = priority_[index], v = priority_[index + 1];
    const int a = owner_[u], b = owner_[v];
    priority_[index] = v; priority_[index + 1] = u;
    position_[v] = index; position_[u] = index + 1;
    poll_periodically();
    if (a == -1 || coefficient(v, a).is_zero()) return;
    if (b == -1) { owner_[u] = -1; owner_[v] = a; }
    else if (a < b) { cancel(b, a, v); owner_[u] = b; owner_[v] = a; }
    else cancel(a, b, v);
  }
  void move_row(int row, int destination, bool debug) {

    int current = position_[row];
    if (current < destination) throw std::logic_error("discovered pivot repeats previous prefix");
    while (current > destination) {
      transpose_unchecked(--current);
      if (debug) assert_valid();
    }

  }
};

}  // namespace integral_native
