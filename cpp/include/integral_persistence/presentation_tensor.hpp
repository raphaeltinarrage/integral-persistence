#pragma once

// Exact Algorithm 7, independent of Python transport.  The inverse-basis
// incidence index visits only rows meeting a relation's sparse boundary.
#include <gmpxx.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace integral_native::presentations_tensor {

using Index = std::size_t;
using Sparse = std::map<Index, mpz_class>;
using Columns = std::vector<Sparse>;
using Simplex = std::vector<Index>;
constexpr Index absent = std::numeric_limits<Index>::max();
using Poll = std::function<void(const char*, Index, Index, Index, bool)>;

struct Input {
    std::vector<Index> vertices;
    std::vector<Index> offsets{0};
    std::vector<Index> sources;
    Index vertex_count = 0;
    Index original_size = 0;

    Simplex simplex(Index i) const {
        return Simplex(vertices.begin() + offsets[i], vertices.begin() + offsets[i + 1]);
    }
};

struct Snapshot {
    Index source = 0;
    Index cycle_rank = 0;
    Index chain_rank = 0;
    Index lower_rank = 0;
    Columns U, W, boundary, complement;
};

struct Output {
    Columns relations;
    std::vector<Index> births, relation_sources;
    std::vector<Snapshot> snapshots;
};

struct SimplexHash {
    Index operator()(const Simplex& simplex) const noexcept {
        Index hash = simplex.size();
        for (Index vertex : simplex)
            hash ^= vertex + Index(0x9e3779b9) + (hash << 6) + (hash >> 2);
        return hash;
    }
};

inline void add_scaled(Sparse& target, const Sparse& source, const mpz_class& scalar,
                       const std::function<void()>& tick) {
    if (scalar == 0) return;
    for (const auto& [row, value] : source) {
        auto [it, inserted] = target.try_emplace(row, 0);
        it->second += scalar * value;
        if (it->second == 0) target.erase(it);
        tick();
    }
}

inline Sparse combine(const Sparse& first, const Sparse& second,
                      const mpz_class& a, const mpz_class& b,
                      const std::function<void()>& tick) {
    Sparse result;
    add_scaled(result, first, a, tick);
    add_scaled(result, second, b, tick);
    return result;
}

struct Bezout { mpz_class g, x, y; };

// Match the reference's pivot-preserving Euclidean choice exactly.  In
// particular, divisibility does not gratuitously exchange basis directions.
inline Bezout bezout(const mpz_class& a, const mpz_class& b,
                     const std::function<void()>& tick) {
    if (a != 0 && mpz_divisible_p(b.get_mpz_t(), a.get_mpz_t()))
        return {abs(a), a > 0 ? 1 : -1, 0};
    mpz_class old_r = abs(a), remainder = abs(b);
    mpz_class old_x = 1, x = 0, old_y = 0, y = 1;
    while (remainder != 0) {
        const mpz_class quotient = old_r / remainder;
        mpz_class next = old_r - quotient * remainder;
        old_r = remainder; remainder = std::move(next);
        next = old_x - quotient * x; old_x = x; x = std::move(next);
        next = old_y - quotient * y; old_y = y; y = std::move(next);
        tick();
    }
    if (old_r == 0) throw std::logic_error("a Bezout operation requires a nonzero pair");
    return {old_r, a >= 0 ? old_x : -old_x, b >= 0 ? old_y : -old_y};
}

inline Columns rows_to_columns(const Columns& rows, Index ncols) {
    Columns result(ncols);
    for (Index row = 0; row < rows.size(); ++row)
        for (const auto& [column, value] : rows[row]) result.at(column)[row] = value;
    return result;
}

inline void assert_basis(const Columns& U, const Columns& W, const Columns& boundary,
                         const Columns& H, Index rank,
                         const std::function<void()>& tick) {
    if (U.size() != W.size() || U.size() != rank + H.size())
        throw std::logic_error("inconsistent tensor basis dimensions");
    std::unordered_set<Index> pivots;
    for (const auto& column : H)
        if (column.empty() || !pivots.insert(column.rbegin()->first).second)
            throw std::logic_error("tensor image pivots are not distinct");
    for (Index j = 0; j < U.size(); ++j) {
        for (Index row = 0; row < W.size(); ++row) {
            mpz_class value = 0;
            for (const auto& [k, coefficient] : U[j]) {
                const auto found = W[row].find(k);
                if (found != W[row].end()) value += coefficient * found->second;
                tick();
            }
            if (value != (row == j ? 1 : 0))
                throw std::logic_error("tensor inverse basis identity failed");
        }
        Sparse image;
        for (const auto& [k, coefficient] : U[j])
            add_scaled(image, boundary.at(k), coefficient, tick);
        if ((j < rank && !image.empty()) || (j >= rank && image != H[j - rank]))
            throw std::logic_error("tensor boundary basis identity failed");
    }
}

// Row IDs remain stable when a newly born cycle is placed before all image
// directions.  Only the separate order vectors change; the sparse incidence
// index never needs a global relabeling.
class InverseRows {
public:
    Columns rows;
    std::vector<std::unordered_set<Index>> incidence;

    void append_identity() {
        const Index i = rows.size();
        rows.push_back(Sparse{{i, 1}});
        incidence.push_back(std::unordered_set<Index>{i});
    }

    void replace(Index i, Sparse replacement, const std::function<void()>& tick) {
        for (const auto& [column, value] : rows[i]) {
            (void) value;
            incidence[column].erase(i);
            tick();
        }
        rows[i] = std::move(replacement);
        for (const auto& [column, value] : rows[i]) {
            (void) value;
            incidence[column].insert(i);
            tick();
        }
    }

    Sparse coordinates(const Sparse& vector, const std::function<void()>& tick) const {
        Sparse result;
        for (const auto& [column, coefficient] : vector) {
            for (Index row : incidence.at(column)) {
                auto [it, inserted] = result.try_emplace(row, 0);
                it->second += rows[row].at(column) * coefficient;
                if (it->second == 0) result.erase(it);
                tick();
            }
        }
        return result;
    }
};

inline Output general(const Input& input, Index degree, bool checked, bool trace,
                      const Poll& poll) {
    Output result;
    Columns U, H, boundary;
    InverseRows inverse;
    std::vector<Index> cycles, complements, cycle_position;
    std::unordered_map<Index, Index> owners;
    std::unordered_map<Simplex, Index, SimplexHash> lower, chain;
    Index completed = 0;
    const auto tick = [&]() {
        poll("construction", completed, input.original_size, cycles.size(), false);
    };
    poll("construction", 0, input.original_size, 0, true);
    for (Index position = 0; position < input.sources.size(); ++position) {
        const Index source = input.sources[position];
        completed = source;
        const Simplex simplex = input.simplex(position);
        const Index dimension = simplex.size() - 1;
        if (degree > 0 && dimension == degree - 1) {
            lower.emplace(simplex, lower.size());
        } else if (dimension == degree) {
            const Index n = chain.size();
            chain.emplace(simplex, n);
            Sparse column;
            if (degree > 0) {
                for (Index i = 0; i < simplex.size(); ++i) {
                    Simplex face = simplex;
                    face.erase(face.begin() + i);
                    column[lower.at(face)] = i % 2 == 0 ? 1 : -1;
                }
            }
            if (checked) boundary.push_back(column);
            inverse.append_identity();
            cycle_position.push_back(absent);
            if (checked) U.push_back(Sparse{{n, 1}});
            while (!column.empty()) {
                const Index pivot = column.rbegin()->first;
                const auto owner = owners.find(pivot);
                if (owner == owners.end()) break;
                const Index i = owner->second;
                const mpz_class a = H[i].at(pivot), b = column.at(pivot);
                const auto [g, x, y] = bezout(a, b, tick);
                const Sparse old_H = std::move(H[i]);
                H[i] = combine(old_H, column, x, y, tick);
                column = combine(old_H, column, -b / g, a / g, tick);
                const Index k = complements[i];
                Sparse first = combine(inverse.rows[k], inverse.rows[n], a / g, b / g, tick);
                Sparse last = combine(inverse.rows[k], inverse.rows[n], -y, x, tick);
                inverse.replace(k, std::move(first), tick);
                inverse.replace(n, std::move(last), tick);
                if (checked) {
                    first = combine(U[k], U[n], x, y, tick);
                    last = combine(U[k], U[n], -b / g, a / g, tick);
                    U[k] = std::move(first);
                    U[n] = std::move(last);
                }
                tick();
            }
            if (column.empty()) {
                cycle_position[n] = cycles.size();
                cycles.push_back(n);
                result.births.push_back(source);
            } else {
                owners.emplace(column.rbegin()->first, H.size());
                H.push_back(std::move(column));
                complements.push_back(n);
            }
        } else if (dimension == degree + 1) {
            Sparse chain_boundary;
            for (Index i = 0; i < simplex.size(); ++i) {
                Simplex face = simplex;
                face.erase(face.begin() + i);
                chain_boundary[chain.at(face)] = i % 2 == 0 ? 1 : -1;
            }
            const Sparse coordinates = inverse.coordinates(chain_boundary, tick);
            Sparse relation;
            for (const auto& [row, value] : coordinates) {
                if (cycle_position[row] == absent)
                    throw std::logic_error("a simplex boundary has nonzero complementary coordinates");
                relation[cycle_position[row]] = value;
            }
            result.relations.push_back(std::move(relation));
            result.relation_sources.push_back(source);
        }

        if (checked) {
            Columns ordered_U, ordered_W;
            ordered_U.reserve(U.size()); ordered_W.reserve(U.size());
            for (const auto& ids : {cycles, complements}) {
                for (Index id : ids) {
                    ordered_U.push_back(U[id]);
                    ordered_W.push_back(inverse.rows[id]);
                    tick();
                }
            }
            assert_basis(ordered_U, ordered_W, boundary, H, cycles.size(), tick);
            if (trace) result.snapshots.push_back(Snapshot{
                source, cycles.size(), U.size(), lower.size(), std::move(ordered_U),
                rows_to_columns(ordered_W, U.size()), boundary, H});
        }
        completed = source + 1;
        tick();
    }
    poll("construction", input.original_size, input.original_size, cycles.size(), true);
    return result;
}

struct ForestEdge { Index u, v; };
struct Adjacent { Index vertex, edge; int sign; };

inline Snapshot forest_snapshot(Index source, const std::vector<Index>& vertices,
                                const std::vector<Index>& vertex_positions,
                                const std::vector<ForestEdge>& edges,
                                const std::vector<std::vector<Adjacent>>& tree,
                                const std::vector<Index>& chords,
                                const std::function<void()>& tick) {
    Columns paths(tree.size());
    std::vector<bool> seen(tree.size(), false), roots(tree.size(), false);
    for (Index root : vertices) {
        if (seen[root]) continue;
        roots[root] = seen[root] = true;
        std::vector<Index> stack{root};
        while (!stack.empty()) {
            const Index vertex = stack.back(); stack.pop_back();
            for (const auto& adjacent : tree[vertex]) {
                if (seen[adjacent.vertex]) continue;
                paths[adjacent.vertex] = paths[vertex];
                paths[adjacent.vertex][adjacent.edge] = adjacent.sign;
                seen[adjacent.vertex] = true;
                stack.push_back(adjacent.vertex);
                tick();
            }
        }
    }
    Snapshot snap;
    snap.source = source; snap.cycle_rank = chords.size();
    snap.chain_rank = edges.size(); snap.lower_rank = vertices.size();
    Columns rows, incidence(tree.size());
    for (Index edge : chords) {
        const auto [u, v] = edges[edge];
        Sparse cycle = combine(paths[u], paths[v], 1, -1, tick);
        cycle[edge] = 1;
        snap.U.push_back(std::move(cycle));
        rows.push_back(Sparse{{edge, 1}});
    }
    for (Index edge = 0; edge < edges.size(); ++edge) {
        const auto [u, v] = edges[edge];
        incidence[u][edge] = -1; incidence[v][edge] = 1;
        snap.boundary.push_back(Sparse{{vertex_positions[u], -1}, {vertex_positions[v], 1}});
    }
    for (Index vertex : vertices) {
        if (roots[vertex]) continue;
        snap.U.push_back(paths[vertex]);
        rows.push_back(incidence[vertex]);
        Sparse image;
        for (const auto& [edge, coefficient] : paths[vertex])
            add_scaled(image, snap.boundary[edge], coefficient, tick);
        snap.complement.push_back(std::move(image));
    }
    assert_basis(snap.U, rows, snap.boundary, snap.complement, chords.size(), tick);
    snap.W = rows_to_columns(rows, edges.size());
    return snap;
}

inline Output forest(const Input& input, bool checked, bool trace, const Poll& poll) {
    Output result;
    std::vector<Index> parents(input.vertex_count, absent), sizes(input.vertex_count, 1);
    std::vector<Index> vertices, vertex_positions(input.vertex_count, absent), chords;
    std::vector<ForestEdge> edges;
    std::vector<std::vector<Adjacent>> tree(checked ? input.vertex_count : 0);
    std::unordered_map<Simplex, Index, SimplexHash> chord_rows;
    Index completed = 0;
    const auto tick = [&]() {
        poll("construction", completed, input.original_size, result.births.size(), false);
    };
    const auto root = [&](Index vertex) {
        while (parents.at(vertex) != vertex) {
            parents[vertex] = parents.at(parents[vertex]);
            vertex = parents[vertex];
            tick();
        }
        return vertex;
    };
    poll("construction", 0, input.original_size, 0, true);
    for (Index position = 0; position < input.sources.size(); ++position) {
        const Index source = input.sources[position];
        completed = source;
        const Simplex simplex = input.simplex(position);
        if (simplex.size() == 1) {
            const Index vertex = simplex.front();
            parents[vertex] = vertex;
            vertex_positions[vertex] = vertices.size();
            vertices.push_back(vertex);
        } else if (simplex.size() == 2) {
            const Index u = simplex[0], v = simplex[1];
            Index first = root(u), second = root(v);
            const Index edge = edges.size();
            if (checked) edges.push_back({u, v});
            if (first != second) {
                if (sizes[first] < sizes[second]) std::swap(first, second);
                parents[second] = first; sizes[first] += sizes[second];
                if (checked) {
                    tree[u].push_back({v, edge, 1});
                    tree[v].push_back({u, edge, -1});
                }
            } else {
                chord_rows.emplace(simplex, result.births.size());
                result.births.push_back(source);
                if (checked) chords.push_back(edge);
            }
        } else if (simplex.size() == 3) {
            Sparse relation;
            for (Index i = 0; i < 3; ++i) {
                Simplex edge = simplex;
                edge.erase(edge.begin() + i);
                const auto row = chord_rows.find(edge);
                if (row != chord_rows.end()) relation[row->second] = i % 2 == 0 ? 1 : -1;
            }
            result.relations.push_back(std::move(relation));
            result.relation_sources.push_back(source);
        }
        if (checked) {
            Snapshot snapshot = forest_snapshot(source, vertices, vertex_positions,
                                                edges, tree, chords, tick);
            if (trace) result.snapshots.push_back(std::move(snapshot));
        }
        completed = source + 1;
        tick();
    }
    poll("construction", input.original_size, input.original_size, result.births.size(), true);
    return result;
}

} // namespace integral_native::presentations_tensor
