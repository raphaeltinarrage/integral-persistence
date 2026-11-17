// Test-only access to both actual upstream sparse insertion paths.
// All prerequisite headers are loaded before exposing private members.
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology/Persistent_cohomology_column.h>
#include <gudhi/Persistent_cohomology/Field_Zp.h>
#include <gudhi/Persistent_cohomology/Multi_field.h>
#include <gudhi/Simple_object_pool.h>
#include <boost/pending/disjoint_sets.hpp>
#include <fstream>
#include <set>
#include <tuple>
#define private public
#include <gudhi/Persistent_cohomology.h>
#undef private
#include <iostream>
#include <stdexcept>

using Tree = Gudhi::Simplex_tree<>;
using Field = Gudhi::persistent_cohomology::Multi_field;
using Persistence = Gudhi::persistent_cohomology::Persistent_cohomology<Tree, Field>;

void initialize(Field& field, const std::vector<mpz_class>& primes) {
  field.init_exact(primes);
}

bool rejected(const std::vector<mpz_class>& primes) {
  try {
    Field field;
    initialize(field, primes);
    return false;
  } catch (const std::invalid_argument&) { return true; }
}

int main() {
  Field field;
  initialize(field, {2, 3});
  bool zero = field.times_minus(2, 3) == 0;
  Tree tree;
  Persistence persistence(tree, true);
  initialize(persistence.coeff_field_, {2, 3});
  Persistence::Column merge(0), tail(0);
  auto* retained = persistence.cell_pool_.construct(Persistence::Cell(2, 1, &merge));
  merge.col_.push_back(*retained);
  Persistence::A_ds_type other{{1, mpz_class(3)}};
  persistence.plus_equal_column(merge, other, mpz_class(2));
  persistence.plus_equal_column(tail, other, mpz_class(2));
  bool merge_ok = std::distance(merge.col_.begin(), merge.col_.end()) == 1;
  bool tail_ok = tail.col_.empty();
  auto dispose = [&](auto* cell) { persistence.cell_pool_.destroy(cell); };
  merge.col_.clear_and_dispose(dispose);
  tail.col_.clear_and_dispose(dispose);
  initialize(field, {2, 3});
  bool reset = field.characteristic() == 6 && field.primes_.size() == 2 && field.Uvect_.size() == 2;
  std::cout << "{\"canonical_zero\":" << (zero ? "true" : "false")
    << ",\"merge_insertion_skips_zero\":" << (merge_ok ? "true" : "false")
    << ",\"tail_insertion_skips_zero\":" << (tail_ok ? "true" : "false")
    << ",\"reinitialization_resets\":" << (reset ? "true" : "false");
  bool algebra = true;
  // A deterministic proven prime > 2^64, with all projectors checked componentwise.
  mpz_class huge("18446744073709551629");
  initialize(field, {2, 3, huge});
  for (std::size_t i = 0; i < field.primes_.size(); ++i)
    for (std::size_t j = 0; j < field.primes_.size(); ++j)
      algebra &= (field.Uvect_[i] % field.primes_[j]) == (i == j ? 1 : 0);
  algebra &= field.multiplicative_identity() == 1 && field.additive_identity() == 0;
  for (const mpz_class& x : std::vector<mpz_class>{mpz_class(0), mpz_class(2), mpz_class(3), mpz_class(-7), huge}) {
    auto canonical = field.plus_times_equal(x, 0, 0);
    algebra &= canonical >= 0 && canonical < field.characteristic();
    algebra &= field.times(canonical, 1) == canonical;
    algebra &= field.plus_equal(canonical, 0) == canonical;
    auto partial = field.inverse(canonical, field.characteristic());
    for (const auto& prime : field.primes_) {
      if (canonical % prime == 0) algebra &= partial.first % prime == 0;
      else algebra &= (partial.first * canonical) % prime == 1;
    }
  }
  initialize(field, {huge});
  algebra &= field.characteristic() == huge && field.Uvect_.size() == 1 && field.Uvect_[0] == 1;
  std::cout << ",\"arbitrary_precision_algebra\":" << (algebra ? "true" : "false")
    << ",\"empty_rejected\":" << (rejected({}) ? "true" : "false")
    << ",\"duplicate_rejected\":" << (rejected({2, 2}) ? "true" : "false")
    << ",\"composite_rejected\":" << (rejected({2, 9}) ? "true" : "false")
    << ",\"large_prime\":\"" << huge << "\"";
  std::cout << ",\"gmp_version\":\"" << gmp_version << "\"}\n";
  return zero && merge_ok && tail_ok && reset && algebra && rejected({})
      && rejected({2, 2}) && rejected({2, 9}) ? 0 : 1;
}
