// Local integration for the optional patched GUDHI 3.6.0 CRT competitor.
// Upstream attribution: ../vendor/LICENSE and ../source_manifest.json.
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/Persistent_cohomology/Multi_field.h>
#include <flint/fmpz.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

using Tree = Gudhi::Simplex_tree<>;
using Field = Gudhi::persistent_cohomology::Multi_field;
using Persistence = Gudhi::persistent_cohomology::Persistent_cohomology<Tree, Field>;
using Clock = std::chrono::steady_clock;
using Interval = std::pair<long long, long long>;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// The patched upstream initializer has an additional GMP composite screen.
// This FLINT call is the proof required before using a characteristic.
void prove_prime(const mpz_class& prime) {
  if (prime < 2) throw std::invalid_argument("characteristic must be prime");
  fmpz_t value;
  fmpz_init(value);
  fmpz_set_mpz(value, prime.get_mpz_t());
  const int proved = fmpz_is_prime(value);
  fmpz_clear(value);
  if (proved != 1) throw std::invalid_argument("characteristic is not a proved prime");
}

int main(int argc, char* argv[]) {
  try {
    bool persistence_dim_max = true;
    if (argc == 2 && std::string(argv[1]) == "--no-top-dimensional-persistence")
      persistence_dim_max = false;
    else if (argc != 1)
      throw std::invalid_argument("unknown command-line option");
    // Text protocol: n k degree; k decimal primes; then n size/vertex rows.
    std::size_t n, k;
    int degree;
    if (!(std::cin >> n >> k >> degree)) throw std::invalid_argument("missing protocol header");
    if (degree < 0) throw std::invalid_argument("degree must be nonnegative");
    if (k == 0) throw std::invalid_argument("empty fields must bypass native computation");
    if (n >= (std::size_t{1} << 53))
      throw std::invalid_argument("serialization grades must be exactly representable as doubles");
    std::vector<mpz_class> primes(k);
    for (auto& prime : primes) {
      std::string decimal;
      if (!(std::cin >> decimal) || prime.set_str(decimal, 10))
        throw std::invalid_argument("invalid decimal characteristic");
    }
    auto start = Clock::now();
    for (std::size_t i = 0; i < k; ++i) {
      prove_prime(primes[i]);
      if (i && primes[i - 1] >= primes[i])
        throw std::invalid_argument("characteristics must be distinct and increasing");
    }
    const double certification = elapsed(start);
    std::vector<std::vector<int>> simplices(n);
    std::set<std::vector<int>> seen;
    for (auto& simplex : simplices) {
      std::size_t size;
      if (!(std::cin >> size) || size == 0)
        throw std::invalid_argument("invalid simplex size");
      simplex.resize(size);
      for (std::size_t j = 0; j < size; ++j) {
        auto& vertex = simplex[j];
        if (!(std::cin >> vertex) || vertex < 0 || vertex == std::numeric_limits<int>::max())
          throw std::invalid_argument("GUDHI vertex labels must fit nonnegative int");
        if (j && simplex[j - 1] >= vertex)
          throw std::invalid_argument("simplex vertices must be distinct and increasing");
      }
      if (seen.count(simplex)) throw std::invalid_argument("duplicate simplex");
      if (size > 1) {
        for (std::size_t j = 0; j < size; ++j) {
          auto face = simplex;
          face.erase(face.begin() + j);
          if (!seen.count(face)) throw std::invalid_argument("every face must precede its coface");
        }
      }
      seen.insert(simplex);
    }
    std::string extra;
    if (std::cin >> extra) throw std::invalid_argument("unexpected trailing protocol data");
    double conversion = 0, initialization = 0, computation = 0, output = 0;
    double output_cleanup = 0, output_sorting = 0, output_json_encoding = 0;
    start = Clock::now();
    Tree tree;
    for (std::size_t i = 0; i < n; ++i) {
      auto inserted = tree.insert_simplex(simplices[i], static_cast<double>(i));
      if (!inserted.second || tree.num_simplices() != i + 1)
        throw std::invalid_argument("conversion changed simplex count/order");
    }
    tree.initialize_filtration();
    if (!persistence_dim_max && degree == tree.dimension())
      throw std::invalid_argument("requested degree is the excluded maximal homology dimension");
    std::size_t index = 0;
    for (auto simplex : tree.filtration_simplex_range()) {
      std::vector<int> vertices;
      for (auto vertex : tree.simplex_vertex_range(simplex)) vertices.push_back(vertex);
      std::sort(vertices.begin(), vertices.end());
      if (tree.filtration(simplex) != static_cast<double>(index) || vertices != simplices.at(index))
        throw std::invalid_argument("native filtration order differs from serialization");
      ++index;
    }
    conversion += elapsed(start);
    // Extraction includes result allocation and exact CRT mask expansion.
    // Cleanup, canonical sorting and transport encoding are separate costs.
    start = Clock::now();
    std::vector<std::vector<Interval>> barcodes(k);
    output += elapsed(start);
    start = Clock::now();
    auto persistence = std::make_unique<Persistence>(tree, persistence_dim_max);
    persistence->coeff_field_.init_exact(primes);
    initialization += elapsed(start);
    start = Clock::now();
    persistence->compute_persistent_cohomology();
    computation += elapsed(start);
    start = Clock::now();
    for (const auto& pair : persistence->get_persistent_pairs()) {
      auto birth = std::get<0>(pair), death = std::get<1>(pair);
      if (tree.dimension(birth) != degree) continue;
      const mpz_class& mask = std::get<2>(pair);
      long long b = static_cast<long long>(tree.filtration(birth));
      long long d = death == tree.null_simplex() ? -1 : static_cast<long long>(tree.filtration(death));
      for (std::size_t j = 0; j < k; ++j)
        if (mpz_divisible_p(mask.get_mpz_t(), primes[j].get_mpz_t())) barcodes[j].emplace_back(b, d);
    }
    output += elapsed(start);
    start = Clock::now();
    persistence.reset();
    output_cleanup += elapsed(start);
    start = Clock::now();
    for (auto& barcode : barcodes) std::sort(barcode.begin(), barcode.end());
    output_sorting += elapsed(start);
    start = Clock::now();
    std::ostringstream encoded;
    encoded << "\"barcodes\":{";
    for (std::size_t j = 0; j < k; ++j) {
      if (j) encoded << ',';
      encoded << '"' << primes[j] << "\":[";
      for (std::size_t i = 0; i < barcodes[j].size(); ++i) {
        if (i) encoded << ',';
        encoded << '[' << barcodes[j][i].first << ',';
        if (barcodes[j][i].second < 0) encoded << "null";
        else encoded << barcodes[j][i].second;
        encoded << ']';
      }
      encoded << ']';
    }
    encoded << '}';
    const std::string encoded_barcodes = encoded.str();
    output_json_encoding += elapsed(start);
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::cout << std::setprecision(17) << "{\"status\":\"success\",\"gmp_version\":\"" << gmp_version
      << "\",\"computed_degrees\":\""
      << (persistence_dim_max ? "all_including_maximal" : "below_maximal")
      << "\",\"order_preserved\":true,"
      << encoded_barcodes << ",\"native_peak_rss_kib\":" << usage.ru_maxrss
      << ",\"prime_validation_seconds\":" << certification
      << ",\"output_scope\":\"native_degree_pair_extraction\""
      << ",\"excluded_output_seconds\":{\"native_cleanup\":" << output_cleanup
      << ",\"native_sorting\":" << output_sorting
      << ",\"native_json_encoding\":" << output_json_encoding << '}'
      << ",\"phases\":{\"input_preparation\":" << conversion << ",\"initialization\":" << initialization
      << ",\"computation\":" << computation << ",\"output\":" << output << "}}\n";
    return 0;
  } catch (const std::bad_alloc&) {
    std::cerr << "memory_limit: std::bad_alloc\n";
    return 3;
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}
