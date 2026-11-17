// Algorithm 8: exact saturation of an increasing relation flag.
// At each prefix, the emitted columns span span_Q(L) intersect Z^r.
// Dependent relations are detected directly in the current quotient lattice.
#pragma once

#include <gmpxx.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace integral_native::presentations_free {

using Index = std::size_t;
using Vector = std::map<Index, mpz_class>;
using Basis = std::vector<Vector>;
// Polling is independent of Python. The binding reacquires the GIL only for
// signals, progress callbacks, input transfer, and requested exact snapshots.
struct Monitor {
  std::function<void(const std::string&, Index, Index, Index, bool)> callback;
  std::string phase;
  Index completed = 0, total = 0, rank = 0;
  Index work = 0;
  void poll() {
    if ((++work & 255) == 0 && callback) callback(phase, completed, total, rank, false);
  }
  void event(bool force = false) {
    if (callback) callback(phase, completed, total, rank, force);
  }
};

inline Vector combination(const Vector& first, const Vector& second,
                          const mpz_class& a, const mpz_class& b,
                          Monitor& monitor) {
  Vector result;
  if (a != 0) {
    for (const auto& [i, value] : first) {
      result.emplace(i, a * value);
      monitor.poll();
    }
  }
  if (b != 0) {
    for (const auto& [i, value] : second) {
      auto found = result.find(i);
      if (found == result.end()) result.emplace(i, b * value);
      else {
        found->second += b * value;
        if (found->second == 0) result.erase(found);
      }
      monitor.poll();
    }
  }
  return result;
}

// The source and target are distinct. Unit-pivot eliminations then visit only
// the source support, instead of copying the entire current target column.
inline void combine_into(Vector& target, const Vector& source,
                         const mpz_class& a, const mpz_class& b,
                         Monitor& monitor) {
  if (a == 0) target.clear();
  else if (a != 1) {
    for (auto& [i, value] : target) { (void)i; value *= a; monitor.poll(); }
  }
  if (b != 0) {
    for (const auto& [i, value] : source) {
      auto [found, inserted] = target.try_emplace(i, 0);
      (void)inserted;
      found->second += b * value;
      if (found->second == 0) target.erase(found);
      monitor.poll();
    }
  }
}

// Match the reference Euclidean choices, including the divisible-pivot shortcut.
inline void bezout(const mpz_class& a, const mpz_class& b,
                   mpz_class& g, mpz_class& x, mpz_class& y,
                   Monitor& monitor) {
  if (a != 0 && mpz_divisible_p(b.get_mpz_t(), a.get_mpz_t())) {
    g = abs(a); x = a > 0 ? 1 : -1; y = 0;
    return;
  }
  mpz_class remainder = abs(b), old_x = 1, next_x = 0;
  mpz_class old_y = 0, next_y = 1;
  g = abs(a);
  while (remainder != 0) {
    mpz_class quotient, next_remainder;
    mpz_fdiv_qr(quotient.get_mpz_t(), next_remainder.get_mpz_t(),
                g.get_mpz_t(), remainder.get_mpz_t());
    g = std::move(remainder); remainder = std::move(next_remainder);
    mpz_class updated_x = old_x - quotient * next_x;
    mpz_class updated_y = old_y - quotient * next_y;
    old_x = std::move(next_x); next_x = std::move(updated_x);
    old_y = std::move(next_y); next_y = std::move(updated_y);
    monitor.poll();
  }
  if (g == 0) throw std::logic_error("zero Bezout pair");
  x = a < 0 ? -old_x : old_x;
  y = b < 0 ? -old_y : old_y;
}

using ReadColumn = std::function<Vector(Index)>;
using Snapshot = std::function<void(Index, Index, const Basis&, const Basis&)>;

struct Result {
  Basis columns;
  std::vector<Index> relation_indices;
};

// Sparse mutable inverse with an incidence index. Multiplication visits only
// rows touching the input support, never every row of the ambient cycle group.
class Inverse {
 public:
  Basis rows;
  std::vector<std::unordered_set<Index>> incidence;

  explicit Inverse(Index n, Monitor& monitor) : rows(n), incidence(n) {
    for (Index i = 0; i < n; ++i) {
      rows[i].emplace(i, 1);
      incidence[i].insert(i);
      monitor.completed = i + 1;
      monitor.poll();
    }
  }

  void replace(Index i, Vector value, Monitor& monitor) {
    for (const auto& [j, entry] : rows[i]) {
      (void)entry;
      incidence[j].erase(i);
      monitor.poll();
    }
    rows[i] = std::move(value);
    for (const auto& [j, entry] : rows[i]) {
      (void)entry;
      incidence[j].insert(i);
      monitor.poll();
    }
  }

  Vector coordinates(const Vector& vector, Index begin, Monitor& monitor) const {
    Vector result;
    for (const auto& [j, coefficient] : vector) {
      for (const Index i : incidence[j]) {
        if (i < begin) continue;
        const mpz_class product = rows[i].at(j) * coefficient;
        auto found = result.find(i);
        if (found == result.end()) result.emplace(i, product);
        else {
          found->second += product;
          if (found->second == 0) result.erase(found);
        }
        monitor.poll();
      }
    }
    return result;
  }

  void combine_into(Index i, Index source, const mpz_class& a,
                    const mpz_class& b, Monitor& monitor) {
    // This is used only with a = +/-1. Scaling preserves the incidence set.
    if (a != 1) {
      for (auto& [j, value] : rows[i]) { (void)j; value *= a; monitor.poll(); }
    }
    if (b != 0) {
      for (const auto& [j, value] : rows[source]) {
        auto [found, inserted] = rows[i].try_emplace(j, 0);
        found->second += b * value;
        if (inserted) incidence[j].insert(i);
        if (found->second == 0) {
          rows[i].erase(found); incidence[j].erase(i);
        }
        monitor.poll();
      }
    }
  }
};

inline Result construct(Index rows, Index count, const ReadColumn& read,
                        bool checked,
                        Monitor& monitor, const Snapshot& snapshot) {
  Result result;
  monitor.phase = "basis_initialization"; monitor.total = rows;
  monitor.completed = monitor.rank = 0; monitor.event(true);
  Basis U(rows);
  for (Index i = 0; i < rows; ++i) {
    U[i].emplace(i, 1);
    monitor.poll();
  }
  Inverse W(rows, monitor);
  monitor.completed = rows; monitor.event(true);
  Index rank = 0;
  monitor.phase = "saturation"; monitor.total = count;
  monitor.completed = monitor.rank = 0; monitor.event(true);
  for (Index j = 0; j < count; ++j) {
    monitor.completed = j;
    Vector column = read(j);
    Vector coordinates;
    if (rank < rows) coordinates = W.coordinates(column, rank, monitor);
    if (!coordinates.empty()) {
      mpz_class pivot = 0;
      const auto current = coordinates.find(rank);
      if (current != coordinates.end()) pivot = current->second;
      for (const auto& [k, value] : coordinates) {
        if (k <= rank) continue;
        mpz_class g, x, y;
        bezout(pivot, value, g, x, y, monitor);
        const mpz_class a = pivot / g, b = value / g;
        if (y == 0) {
          // The pivot divides this coordinate: x = a = sign(pivot).
          // One inverse row and one forward column stay fixed up to sign.
          // Avoid copying/reindexing them, especially when their integers
          // have grown large. This is exactly the reference Bezout step.
          combine_into(U[rank], U[k], a, b, monitor);
          W.combine_into(k, rank, a, -b, monitor);
          if (x < 0) {
            for (auto& [i, entry] : W.rows[rank]) { (void)i; entry = -entry; monitor.poll(); }
            for (auto& [i, entry] : U[k]) { (void)i; entry = -entry; monitor.poll(); }
          }
        } else {
          Vector next_row = combination(W.rows[rank], W.rows[k], x, y, monitor);
          Vector other_row = combination(W.rows[rank], W.rows[k], -b, a, monitor);
          Vector next_column = combination(U[rank], U[k], a, b, monitor);
          Vector other_column = combination(U[rank], U[k], -y, x, monitor);
          U[rank] = std::move(next_column); U[k] = std::move(other_column);
          W.replace(rank, std::move(next_row), monitor);
          W.replace(k, std::move(other_row), monitor);
        }
        pivot = std::move(g);
        monitor.poll();
      }
      if (pivot < 0) {
        for (auto& [i, value] : U[rank]) { (void)i; value = -value; monitor.poll(); }
        for (auto& [i, value] : W.rows[rank]) { (void)i; value = -value; monitor.poll(); }
      }
      result.relation_indices.push_back(j);
      if (!checked) W.replace(rank, {}, monitor);
      ++rank;
    }
    if (checked) {
      // WU = I certifies integrality and unimodularity. Operations touch only
      // active coordinates, so previously appended relation columns are fixed.
      for (Index k = 0; k < rows; ++k) {
        const auto inverse = W.coordinates(U[k], 0, monitor);
        if (inverse.size() != 1 || inverse.begin()->first != k || inverse.begin()->second != 1)
          throw std::logic_error("native free basis lost its exact inverse");
      }
      if (!W.coordinates(column, rank, monitor).empty())
        throw std::logic_error("input relation lies outside its saturated prefix");
      if (snapshot) snapshot(j, rank, U, W.rows);
    }
    monitor.completed = j + 1; monitor.rank = rank; monitor.event();
    if (!checked && rank == rows) {
      // A full-rank saturated sublattice is Z^r. Every remaining quotient is zero.
      break;
    }
  }
  monitor.completed = count; monitor.rank = rank; monitor.event(true);
  U.resize(rank);
  result.columns = std::move(U);
  return result;
}

}  // namespace integral_native::presentations_free
