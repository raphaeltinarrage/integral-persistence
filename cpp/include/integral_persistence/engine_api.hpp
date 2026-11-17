#pragma once

#include <gmpxx.h>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace integral_native {

using KernelColumns = std::vector<std::vector<std::pair<int, mpz_class>>>;
using KernelPoll = std::function<void()>;

class NativeEngine;
struct NativeGroup {
  std::vector<std::string> primes;
  std::shared_ptr<NativeEngine> state;
};
struct GroupResult { std::vector<NativeGroup> groups; };

/** Internal exact reduction interface, with a single production backend.
 *
 * Rows 0,...,m-1 are the implicit identity block, and rows m,...,m+n-1
 * are the original matrix. Row priorities list the highest priority first.
 * These native interfaces are implementation details, not a public C++ API.
 */
class NativeEngine : public std::enable_shared_from_this<NativeEngine> {
 public:
  virtual ~NativeEngine() = default;
  virtual void reduce(bool checked = false) = 0;
  virtual void exhaustive(bool checked = false) = 0;
  virtual void transpose(int index, bool checked = false) = 0;
  virtual void correct_priority(const std::vector<int>& target, bool checked = false) = 0;
  virtual void correct_prime(const std::string& prime, bool checked = false) = 0;
  virtual GroupResult correct_groups(const std::vector<std::string>& primes,
                                    bool checked = false) = 0;
  virtual std::shared_ptr<NativeEngine> fork() const = 0;
  virtual std::vector<int> pivots() const = 0;
  virtual std::vector<std::string> minors() const = 0;
  virtual const std::vector<int>& priority() const = 0;
  virtual const std::vector<int>& owner() const = 0;
  virtual int nrows() const = 0;
  virtual int ncols() const = 0;
  virtual void set_poll(KernelPoll poll) = 0;
  virtual void assert_valid() const = 0;
  virtual void assert_product(const KernelColumns& original) const = 0;
};

inline mpz_class parse_integer(const std::string& text) {
  if (text.empty() || (text[0] == '-' && text.size() == 1))
    throw std::invalid_argument("empty exact integer");
  for (std::size_t i = text[0] == '-' ? 1 : 0; i < text.size(); ++i)
    if (text[i] < '0' || text[i] > '9')
      throw std::invalid_argument("expected a decimal exact integer");
  return mpz_class(text, 10);
}

}  // namespace integral_native
