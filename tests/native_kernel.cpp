/** Independent dense oracles for the native sparse reduction kernel.
 *
 * The oracle scans complete dense columns and performs elementary field
 * operations directly. It never calls production column operations.
 */
#include "integral_persistence/lazy_engine.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <sstream>

using integral_native::KernelColumns;
using integral_native::LazyCompactDenseEngine;
using RationalMatrix = std::vector<std::vector<mpq_class>>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<int> canonical_order(int rows, int columns) {
  std::vector<int> order(rows + columns);
  std::iota(order.rbegin(), order.rend(), 0);
  return order;
}

RationalMatrix augmented(const KernelColumns& input, int rows) {
  const auto columns = input.size();
  RationalMatrix result(columns, std::vector<mpq_class>(rows + columns));
  for (std::size_t j = 0; j < columns; ++j) {
    result[j][j] = 1;
    for (const auto& [row, value] : input[j]) result[j][columns + row] = value;
  }
  return result;
}

std::vector<int> rational_oracle(RationalMatrix& columns, const std::vector<int>& order) {
  std::vector<int> owner(order.size(), -1), pivots;
  for (std::size_t j = 0; j < columns.size(); ++j) {
    auto pivot = [&]() {
      for (int row : order) if (columns[j][row] != 0) return row;
      throw std::logic_error("dense augmented column vanished");
    };
    int row = pivot();
    while (owner[row] != -1) {
      const auto source = static_cast<std::size_t>(owner[row]);
      mpq_class factor = columns[j][row] / columns[source][row];
      for (std::size_t i = 0; i < order.size(); ++i)
        columns[j][i] -= factor * columns[source][i];
      row = pivot();
    }
    owner[row] = static_cast<int>(j);
    pivots.push_back(row);
  }
  return pivots;
}

std::vector<int> modular_oracle(const KernelColumns& input, int rows, int prime) {
  const int count = static_cast<int>(input.size()), total = count + rows;
  std::vector<std::vector<int>> columns(count, std::vector<int>(total));
  for (int j = 0; j < count; ++j) {
    columns[j][j] = 1;
    for (const auto& [row, value] : input[j])
      columns[j][count + row] = static_cast<int>(mpz_fdiv_ui(value.get_mpz_t(), prime));
  }
  std::vector<int> owner(total, -1), pivots;
  for (int j = 0; j < count; ++j) {
    auto pivot = [&]() {
      for (int row = total; row-- > 0;) if (columns[j][row]) return row;
      throw std::logic_error("dense augmented modular column vanished");
    };
    int row = pivot();
    while (owner[row] != -1) {
      const int source = owner[row];
      int inverse = 1;
      while ((inverse * columns[source][row]) % prime != 1) ++inverse;
      const int factor = (columns[j][row] * inverse) % prime;
      for (int i = 0; i < total; ++i) {
        const int value = (columns[j][i] - factor * columns[source][i]) % prime;
        columns[j][i] = value < 0 ? value + prime : value;
      }
      row = pivot();
    }
    owner[row] = j;
    pivots.push_back(row);
  }
  return pivots;
}

void check_matrix(const KernelColumns& input, int rows, std::mt19937& random) {
  auto owned = input;
  std::size_t polls = 0;
  auto engine = std::make_shared<LazyCompactDenseEngine>(rows, std::move(owned), [&] { ++polls; });
  require(owned.empty(), "input storage was not consumed");
  engine->reduce(true);
  engine->assert_product(input);
  auto dense = augmented(input, rows);
  const auto canonical = canonical_order(rows, static_cast<int>(input.size()));
  const auto generic = rational_oracle(dense, canonical);
  require(engine->pivots() == generic, "canonical pivots disagree with dense oracle");
  const auto minors = engine->minors();
  mpq_class minor = 1;
  for (std::size_t j = 0; j < input.size(); ++j) {
    minor *= dense[j][generic[j]];
    require(minor.get_den() == 1 && minor != 0, "dense signed minor is not integral");
    require(minors[j] == minor.get_num().get_str(), "signed minor lost its scale or sign");
  }

  // Algorithm 2: random adjacent transpositions, with the parent left intact.
  auto child = engine->fork();
  auto order = canonical;
  if (order.size() > 1) {
    for (int step = 0; step < 8; ++step) {
      int index = static_cast<int>(random() % (order.size() - 1));
      std::swap(order[index], order[index + 1]);
      child->transpose(index, true);
      child->assert_product(input);
      auto fresh = augmented(input, rows);
      require(child->pivots() == rational_oracle(fresh, order),
              "adjacent priority update disagrees with dense oracle");
      require(engine->pivots() == generic, "fork changed its parent's pivots");
      engine->assert_valid();
      engine->assert_product(input);
    }
  }
  // Algorithm 3: a full unrelated priority correction.
  std::shuffle(order.begin(), order.end(), random);
  child->correct_priority(order, true);
  child->assert_product(input);
  auto reordered = augmented(input, rows);
  require(child->pivots() == rational_oracle(reordered, order),
          "full priority correction disagrees with dense oracle");

  engine->exhaustive(true);
  engine->assert_product(input);
  const auto exhaustive_pivots = engine->pivots();
  require(exhaustive_pivots == generic, "exhaustive continuation changed generic pivots");
  auto exhaustive_dense = augmented(input, rows);
  require(rational_oracle(exhaustive_dense, engine->priority()) == generic,
          "exhaustive pivot priority is inconsistent");

  // Algorithms 4-5: independently corrected primes and prefix-sharing groups.
  const std::vector<std::string> primes = {"2", "3", "5", "7", "11"};
  for (const auto& prime : primes) {
    auto independent = engine->fork();
    independent->correct_prime(prime, true);
    independent->assert_product(input);
    require(independent->pivots() == modular_oracle(input, rows, std::stoi(prime)),
            "prime correction disagrees with modular oracle");
    require(engine->pivots() == exhaustive_pivots, "prime correction mutated its exhaustive parent");
  }
  auto groups = engine->correct_groups(primes, true);
  std::set<std::string> seen;
  std::set<std::vector<int>> profiles;
  for (const auto& group : groups.groups) {
    group.state->assert_valid();
    group.state->assert_product(input);
    require(profiles.insert(group.state->pivots()).second,
            "distinct groups have identical entire pivot prefixes");
    for (const auto& prime : group.primes) {
      require(seen.insert(prime).second, "a prime appears twice in grouped correction");
      require(group.state->pivots() == modular_oracle(input, rows, std::stoi(prime)),
              "grouped correction disagrees with modular oracle");
    }
  }
  require(seen.size() == primes.size(), "grouped correction omitted a prime");
  if (!input.empty()) require(polls != 0, "native polling was not reached");
}

int main() {
  try {
    std::mt19937 random(1937401);
    for (int test = 0; test < 180; ++test) {
      const int rows = static_cast<int>(random() % 8), columns = static_cast<int>(random() % 9);
      KernelColumns input(columns);
      for (auto& column : input) for (int row = 0; row < rows; ++row) {
        long value = static_cast<long>(random() % 37) - 18;
        if (value && random() % 4) column.emplace_back(row, value);
      }
      check_matrix(input, rows, random);
    }
    // Divergent prime prefixes, repeated columns and dependent signed minors.
    check_matrix({{{0, 2}, {1, 3}}, {{0, 4}, {1, 9}}, {{0, 6}, {1, 12}}}, 2, random);
    // Word boundaries force rollback and exact promotion; huge source entries
    // additionally exercise the no-copy GMP adoption and rational fallback.
    const mpz_class max_word("9223372036854775807"), min_word("-9223372036854775808");
    const mpz_class huge = (mpz_class(1) << 521) + 17;
    const std::vector<mpz_class> values = {0, 1, -1, 2, -3, max_word, min_word, huge, -huge};
    for (int test = 0; test < 24; ++test) {
      KernelColumns input(5);
      for (auto& column : input) for (int row = 0; row < 4; ++row) {
        auto value = values[random() % values.size()];
        if (value != 0) column.emplace_back(row, std::move(value));
      }
      check_matrix(input, 4, random);
    }
    // A poll exception propagates; no completed result is fabricated.
    bool interrupted = false;
    auto input = KernelColumns{{{0, 1}}};
    auto cancellable = std::make_shared<LazyCompactDenseEngine>(1, std::move(input));
    cancellable->set_poll([] { throw std::runtime_error("cancelled"); });
    try { cancellable->reduce(); } catch (const std::runtime_error&) { interrupted = true; }
    require(interrupted, "poll exception was swallowed");
    std::cout << "native kernel invariants and independent dense oracles passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
