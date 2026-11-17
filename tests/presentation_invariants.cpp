// Independent dense checks of Algorithms 7 and 8. No production column
// arithmetic is used by the rank, determinant, product or boundary oracles.
#include "integral_persistence/presentation_free.hpp"
#include "integral_persistence/presentation_tensor.hpp"

#include <iostream>
#include <numeric>
#include <random>
#include <set>

namespace {
using namespace integral_native;
using Index = std::size_t;
using Dense = std::vector<std::vector<mpq_class>>;
using SparseColumns = presentations_free::Basis;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Dense dense(const SparseColumns& columns, Index rows, Index count) {
    Dense result(rows, std::vector<mpq_class>(count));
    for (Index j = 0; j < count; ++j)
        for (const auto& [i, value] : columns[j]) result.at(i).at(j) = value;
    return result;
}

Index rank(Dense matrix) {
    if (matrix.empty()) return 0;
    const Index columns = matrix.front().size();
    Index pivot_row = 0;
    for (Index j = 0; j < columns && pivot_row < matrix.size(); ++j) {
        Index i = pivot_row;
        while (i < matrix.size() && matrix[i][j] == 0) ++i;
        if (i == matrix.size()) continue;
        std::swap(matrix[i], matrix[pivot_row]);
        const mpq_class pivot = matrix[pivot_row][j];
        for (Index k = j; k < columns; ++k) matrix[pivot_row][k] /= pivot;
        for (i = pivot_row + 1; i < matrix.size(); ++i) {
            const mpq_class scalar = matrix[i][j];
            for (Index k = j; k < columns; ++k)
                matrix[i][k] -= scalar * matrix[pivot_row][k];
        }
        ++pivot_row;
    }
    return pivot_row;
}

mpq_class determinant(Dense matrix) {
    mpq_class result = 1;
    for (Index j = 0; j < matrix.size(); ++j) {
        Index i = j;
        while (i < matrix.size() && matrix[i][j] == 0) ++i;
        if (i == matrix.size()) return 0;
        if (i != j) { std::swap(matrix[i], matrix[j]); result = -result; }
        const mpq_class pivot = matrix[j][j];
        result *= pivot;
        for (i = j + 1; i < matrix.size(); ++i) {
            const mpq_class scalar = matrix[i][j] / pivot;
            for (Index k = j + 1; k < matrix.size(); ++k)
                matrix[i][k] -= scalar * matrix[j][k];
        }
    }
    return result;
}

// A rank-r lattice is saturated iff the gcd of its r-row minors is one.
bool saturated(const Dense& matrix, Index columns) {
    if (columns == 0) return true;
    mpz_class content = 0;
    std::vector<Index> selected;
    const std::function<void(Index)> visit = [&](Index next) {
        if (content == 1) return;
        if (selected.size() == columns) {
            Dense minor;
            for (Index row : selected) minor.push_back(matrix[row]);
            const mpq_class value = determinant(std::move(minor));
            require(value.get_den() == 1, "nonintegral maximal minor");
            const mpz_class numerator = value.get_num();
            mpz_gcd(content.get_mpz_t(), content.get_mpz_t(), numerator.get_mpz_t());
            return;
        }
        for (Index row = next; row < matrix.size(); ++row) {
            selected.push_back(row); visit(row + 1); selected.pop_back();
        }
    };
    visit(0);
    return content == 1;
}

void verify_free(const SparseColumns& input, Index rows) {
    presentations_free::Monitor monitor;
    SparseColumns previous;
    Index previous_rank = 0;
    const auto snapshot = [&](Index j, Index current_rank,
                              const SparseColumns& U, const SparseColumns& W) {
        Dense inverse(rows, std::vector<mpq_class>(rows));
        for (Index i = 0; i < rows; ++i)
            for (const auto& [j, value] : W[i]) inverse[i][j] = value;
        const Dense basis = dense(U, rows, rows);
        for (Index i = 0; i < rows; ++i) for (Index k = 0; k < rows; ++k) {
            mpq_class product = 0;
            for (Index l = 0; l < rows; ++l) product += inverse[i][l] * basis[l][k];
            require(product == (i == k ? 1 : 0), "free inverse identity");
        }
        const Dense relations = dense(input, rows, j + 1);
        const Dense saturated_basis = dense(U, rows, current_rank);
        require(rank(relations) == current_rank, "free prefix rational rank");
        Dense joined = relations;
        for (Index i = 0; i < rows; ++i)
            joined[i].insert(joined[i].end(), saturated_basis[i].begin(), saturated_basis[i].end());
        require(rank(joined) == current_rank, "free prefix rational span");
        require(saturated(saturated_basis, current_rank), "free prefix lattice saturation");
        for (Index i = 0; i < previous_rank; ++i)
            require(previous[i] == U[i], "previous saturated relation changed");
        previous = U;
        previous_rank = current_rank;
    };
    const auto read = [&](Index j) { return input[j]; };
    const auto checked = presentations_free::construct(rows, input.size(), read, true, monitor, snapshot);
    const auto production = presentations_free::construct(rows, input.size(), read, false, monitor, {});
    require(checked.columns == production.columns, "free optimized-path columns differ");
    require(checked.relation_indices == production.relation_indices, "free optimized-path source indices differ");
}

using Simplex = std::vector<Index>;

std::vector<Simplex> closure(const std::vector<Simplex>& facets) {
    std::set<Simplex> faces;
    for (const auto& facet : facets) {
        for (unsigned mask = 1; mask < (1u << facet.size()); ++mask) {
            Simplex face;
            for (Index i = 0; i < facet.size(); ++i)
                if (mask & (1u << i)) face.push_back(facet[i]);
            faces.insert(std::move(face));
        }
    }
    std::vector<Simplex> result(faces.begin(), faces.end());
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.size() == b.size() ? a < b : a.size() < b.size();
    });
    return result;
}

presentations_tensor::Input pack(const std::vector<Simplex>& simplices) {
    presentations_tensor::Input result;
    for (Index j = 0; j < simplices.size(); ++j) {
        result.vertices.insert(result.vertices.end(), simplices[j].begin(), simplices[j].end());
        for (Index vertex : simplices[j]) result.vertex_count = std::max(result.vertex_count, vertex + 1);
        result.offsets.push_back(result.vertices.size());
        result.sources.push_back(j);
    }
    result.original_size = simplices.size();
    return result;
}

void verify_tensor(const std::vector<Simplex>& simplices, Index degree, bool forest) {
    const auto poll = [](const char*, Index, Index, Index, bool) {};
    const auto packed = pack(simplices);
    const auto output = forest
        ? presentations_tensor::forest(packed, true, true, poll)
        : presentations_tensor::general(packed, degree, true, true, poll);
    const auto ordinary = forest
        ? presentations_tensor::forest(packed, false, false, poll)
        : presentations_tensor::general(packed, degree, false, false, poll);
    require(output.relations == ordinary.relations, "tensor optimized-path relations differ");
    require(output.births == ordinary.births, "tensor optimized-path births differ");
    Index relation = 0;
    SparseColumns previous_cycles;
    for (const auto& state : output.snapshots) {
        std::map<Simplex, Index> chains, lower;
        for (Index i = 0; i <= state.source; ++i) {
            if (simplices[i].size() == degree + 1) chains.emplace(simplices[i], chains.size());
            if (degree && simplices[i].size() == degree) lower.emplace(simplices[i], lower.size());
        }
        const Index n = chains.size();
        for (Index i = 0; i < previous_cycles.size(); ++i)
            require(previous_cycles[i] == state.U[i], "previous integral cycle changed");
        previous_cycles.assign(state.U.begin(), state.U.begin() + state.cycle_rank);
        Dense boundary(lower.size(), std::vector<mpq_class>(n));
        if (degree) for (const auto& [simplex, j] : chains) {
            for (Index omitted = 0; omitted < simplex.size(); ++omitted) {
                auto face = simplex;
                face.erase(face.begin() + static_cast<std::ptrdiff_t>(omitted));
                boundary[lower.at(face)][j] = omitted % 2 ? -1 : 1;
            }
        }
        require(boundary == dense(state.boundary, lower.size(), n), "tensor boundary oracle mismatch");
        require(state.cycle_rank + rank(boundary) == n, "tensor integral kernel rank");
        const Dense U = dense(state.U, n, n), W = dense(state.W, n, n);
        for (Index i = 0; i < n; ++i) for (Index j = 0; j < n; ++j) {
            mpq_class product = 0;
            for (Index k = 0; k < n; ++k) product += W[i][k] * U[k][j];
            require(product == (i == j ? 1 : 0), "tensor basis inverse");
        }
        for (Index i = 0; i < lower.size(); ++i) for (Index j = 0; j < state.cycle_rank; ++j) {
            mpq_class product = 0;
            for (Index k = 0; k < n; ++k) product += boundary[i][k] * U[k][j];
            require(product == 0, "tensor cycle has nonzero boundary");
        }
        const auto& simplex = simplices[state.source];
        if (simplex.size() == degree + 2) {
            std::vector<mpq_class> boundary_column(n);
            for (Index omitted = 0; omitted < simplex.size(); ++omitted) {
                auto face = simplex;
                face.erase(face.begin() + static_cast<std::ptrdiff_t>(omitted));
                boundary_column[chains.at(face)] = omitted % 2 ? -1 : 1;
            }
            for (Index i = 0; i < n; ++i) {
                mpq_class coordinate = 0;
                for (Index k = 0; k < n; ++k) coordinate += W[i][k] * boundary_column[k];
                const auto found = output.relations[relation].find(i);
                const mpz_class expected = found == output.relations[relation].end() ? mpz_class(0) : found->second;
                require(coordinate == expected, "tensor relation coordinates");
            }
            require(output.relation_sources[relation] == state.source, "tensor relation source");
            ++relation;
        }
    }
    require(relation == output.relations.size(), "tensor relation count");
}
}  // namespace

int main() {
    std::mt19937 random(715);
    for (Index rows = 0; rows <= 4; ++rows) for (Index count = 0; count <= 6; ++count) {
        for (unsigned repetition = 0; repetition < 8; ++repetition) {
            SparseColumns columns(count);
            for (auto& column : columns) for (Index i = 0; i < rows; ++i) {
                int value = static_cast<int>(random() % 17) - 8;
                if (value) column[i] = value;
            }
            verify_free(columns, rows);
        }
    }
    const mpz_class huge = (mpz_class(1) << 200) + 3;
    verify_free({{{0, -huge}, {1, 6}}, {{0, huge * 2}, {1, -12}}, {{1, huge}, {2, 5}}, {}}, 4);
    const std::vector<Simplex> rp2{{0,1,2},{0,1,3},{0,2,4},{0,3,5},{0,4,5},
                                  {1,2,5},{1,3,4},{1,4,5},{2,3,4},{2,3,5}};
    auto cone = rp2;
    for (auto facet : rp2) { facet.push_back(6); cone.push_back(std::move(facet)); }
    for (const auto& facets : {rp2, cone, std::vector<Simplex>{{0,1,2,3},{1,2,3,4}}}) {
        const auto simplices = closure(facets);
        for (Index degree = 0; degree <= 4; ++degree) verify_tensor(simplices, degree, false);
        verify_tensor(simplices, 1, true);
    }
    std::cout << "presentation prefix lattice and chain-basis invariants passed\n";
}
