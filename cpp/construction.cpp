#include "integral_persistence/api.hpp"
#include "integral_persistence/presentation_free.hpp"
#include "integral_persistence/presentation_tensor.hpp"

#include <set>
#include <unordered_map>

namespace integral_native {
namespace {

using Index = presentations_tensor::Index;

struct SimplexHash {
    std::size_t operator()(const std::vector<int>& simplex) const noexcept {
        std::size_t hash = simplex.size();
        for (const int vertex : simplex)
            hash ^= static_cast<std::size_t>(vertex) + 0x9e3779b9u +
                    (hash << 6) + (hash >> 2);
        return hash;
    }
};

/** Move exact coefficients out of the presentation's sparse map storage. */
IntegerColumns transfer_columns(std::vector<std::map<Index, mpz_class>>& input) {
    IntegerColumns columns;
    columns.reserve(input.size());
    for (auto& source : input) {
        columns.emplace_back();
        auto& column = columns.back();
        column.reserve(source.size());
        for (auto& [row, value] : source)
            column.emplace_back(static_cast<int>(row), std::move(value));
    }
    return columns;
}

std::vector<int> source_indices(const std::vector<Index>& input) {
    std::vector<int> result;
    result.reserve(input.size());
    for (const Index index : input) result.push_back(static_cast<int>(index));
    return result;
}

std::shared_ptr<MatrixData> metadata(const PreparedFiltration& input,
                                     const std::string& module, int degree) {
    auto result = std::make_shared<MatrixData>();
    result->module = module;
    result->degree = degree;
    result->source_count = input.simplices.size();
    result->times = input.times;
    return result;
}

/** Pack only dimensions d-1, d and d+1, retaining original source indices.
 * Numeric grades play no role in the presentation algorithms themselves.
 */
presentations_tensor::Input tensor_input(const PreparedFiltration& input,
                                         int degree) {
    presentations_tensor::Input packed;
    packed.vertex_count = input.vertex_count;
    packed.original_size = input.simplices.size();
    const Index d = static_cast<Index>(degree);
    for (Index source = 0; source < input.simplices.size(); ++source) {
        const auto& simplex = input.simplices[source];
        const Index dimension = simplex.size() - 1;
        if (dimension != d && dimension != d + 1 &&
            !(d > 0 && dimension == d - 1)) continue;
        for (const int vertex : simplex) packed.vertices.push_back(vertex);
        packed.offsets.push_back(packed.vertices.size());
        packed.sources.push_back(source);
    }
    return packed;
}

}  // namespace

BuiltMatrices build_matrices(const PreparedFiltration& input,
                            const std::vector<int>& degrees,
                            const std::vector<std::string>& modules,
                            const Progress& progress) {
    const std::set<int> requested_degrees(degrees.begin(), degrees.end());
    const std::set<std::string> requested_modules(modules.begin(), modules.end());
    for (const int degree : requested_degrees)
        if (degree < 0) throw std::invalid_argument("homology degrees must be nonnegative");
    for (const auto& module : requested_modules)
        if (module != "field" && module != "tensor" && module != "free")
            throw std::invalid_argument("modules must be 'field', 'tensor' or 'free'");
    if (input.simplices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("filtration exceeds the supported simplex index range");
    BuiltMatrices result;
    if (requested_degrees.empty() || requested_modules.empty()) return result;

    if (requested_modules.count("field")) {
        const auto started = Clock::now();
        const auto maximum_dimension = static_cast<std::size_t>(*requested_degrees.rbegin()) + 1;
        std::vector<int> sources, dimensions;
        std::unordered_map<std::vector<int>, int, SimplexHash> indices;
        IntegerColumns boundary;
        report(progress, "field_construction", 0, input.simplices.size(), true);
        for (std::size_t source = 0; source < input.simplices.size(); ++source) {
            const auto& simplex = input.simplices[source];
            const std::size_t dimension = simplex.size() - 1;
            if (dimension > maximum_dimension) continue;
            const int column_index = static_cast<int>(boundary.size());
            if (column_index >= std::numeric_limits<int>::max() / 2)
                throw std::invalid_argument("augmented boundary matrix exceeds the supported index range");
            sources.push_back(static_cast<int>(source));
            dimensions.push_back(static_cast<int>(dimension));
            boundary.emplace_back();
            auto& column = boundary.back();
            if (dimension > 0) {
                column.reserve(simplex.size());
                for (std::size_t omitted = 0; omitted < simplex.size(); ++omitted) {
                    auto face = simplex;
                    face.erase(face.begin() + static_cast<std::ptrdiff_t>(omitted));
                    column.emplace_back(indices.at(face), omitted % 2 ? -1 : 1);
                }
                std::sort(column.begin(), column.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            indices.emplace(simplex, column_index);
            report(progress, "field_construction", source + 1, input.simplices.size());
        }
        const int rows = static_cast<int>(boundary.size());
        auto matrix = std::make_shared<Matrix>(rows, std::move(boundary));
        const double seconds = elapsed(started);
        auto field = std::make_shared<const FieldMetadata>(FieldMetadata{std::move(dimensions), std::move(sources)});
        for (const int degree : requested_degrees) {
            auto data = metadata(input, "field", degree);
            data->matrix = matrix;
            data->field = field;
            data->timings["field_construction"] = seconds;
            result[degree]["field"] = std::move(data);
        }
        report(progress, "field_construction", input.simplices.size(), input.simplices.size(), true);
    }

    if (!requested_modules.count("tensor") && !requested_modules.count("free")) return result;
    for (const int degree : requested_degrees) {
        auto started = Clock::now();
        const auto packed = tensor_input(input, degree);
        const auto poll = [&](const char*, Index completed, Index total, Index, bool force) {
            report(progress, "tensor_construction", completed, total, force);
        };
        auto tensor = degree == 1
            ? presentations_tensor::forest(packed, false, false, poll)
            : presentations_tensor::general(packed, degree, false, false, poll);
        const double tensor_seconds = elapsed(started);
        const auto births = source_indices(tensor.births);
        const auto relation_sources = source_indices(tensor.relation_sources);

        if (requested_modules.count("free")) {
            started = Clock::now();
            presentations_free::Monitor monitor;
            monitor.callback = [&](const std::string&, Index completed, Index total, Index, bool force) {
                report(progress, "free_construction", completed, total, force);
            };
            const auto read = [&](Index column) { return tensor.relations[column]; };
            auto free = presentations_free::construct(tensor.births.size(), tensor.relations.size(),
                                                       read, false, monitor, {});
            auto data = metadata(input, "free", degree);
            data->birth_sources = births;
            data->relation_sources.reserve(free.relation_indices.size());
            for (const Index index : free.relation_indices)
                data->relation_sources.push_back(relation_sources[index]);
            data->matrix = std::make_shared<Matrix>(static_cast<int>(births.size()),
                                                   transfer_columns(free.columns));
            data->timings["tensor_construction"] = tensor_seconds;
            data->timings["free_construction"] = elapsed(started);
            result[degree]["free"] = std::move(data);
        }

        if (requested_modules.count("tensor")) {
            auto data = metadata(input, "tensor", degree);
            data->birth_sources = births;
            data->relation_sources = relation_sources;
            data->matrix = std::make_shared<Matrix>(static_cast<int>(births.size()),
                                                   transfer_columns(tensor.relations));
            data->timings["tensor_construction"] = tensor_seconds;
            result[degree]["tensor"] = std::move(data);
        }
    }
    return result;
}

}  // namespace integral_native
