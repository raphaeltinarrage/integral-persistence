#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "integral_persistence/decoding.hpp"
#include "integral_persistence/barcode_output.hpp"
#include <set>

namespace nb = nanobind;
using namespace nb::literals;
using namespace integral_native;

namespace {
mpz_class integer(nb::handle object) {
    if (PyBool_Check(object.ptr()) || !PyIndex_Check(object.ptr()))
        throw nb::type_error("expected an exact integer (not bool or float)");
    auto value = nb::steal<nb::object>(PyNumber_Index(object.ptr()));
    if (!value.is_valid()) throw nb::python_error();
    auto text = nb::steal<nb::object>(PyNumber_ToBase(value.ptr(), 16));
    if (!text.is_valid()) throw nb::python_error();
    const char* encoded = PyUnicode_AsUTF8(text.ptr());
    if (!encoded) throw nb::python_error();
    mpz_class result;
    if (result.set_str(encoded, 0) != 0) throw nb::value_error("invalid exact integer");
    return result;
}
int index(nb::handle object) {
    auto value = integer(object);
    if (value < 0 || value > std::numeric_limits<int>::max())
        throw nb::value_error("index must be a nonnegative supported integer");
    return static_cast<int>(value.get_si());
}
nb::object py_integer(const mpz_class& value) {
    std::string text = value.get_str(16);
    auto result = nb::steal<nb::object>(PyLong_FromString(text.data(), nullptr, 16));
    if (!result.is_valid()) throw nb::python_error();
    return result;
}
nb::list py_integers(const std::vector<mpz_class>& values) {
    nb::list result;
    for (const auto& value : values) result.append(py_integer(value));
    return result;
}
Grade grade(nb::handle object) {
    if (PyFloat_Check(object.ptr())) return Grade(PyFloat_AS_DOUBLE(object.ptr()));
    if (PyIndex_Check(object.ptr()) && !PyBool_Check(object.ptr())) return Grade(integer(object));
    throw nb::type_error("times must be integers or finite floats; rational objects are unsupported");
}
Grades grades(nb::handle input) {
    Grades result;
    for (auto value : nb::iter(input)) result.push_back(grade(value));
    return result;
}
nb::object py_grade(const Grade& value) {
    if (value.integral) return py_integer(value.integer);
    return nb::float_(value.floating);
}
nb::list py_grades(const Grades& values) {
    nb::list result;
    for (const auto& value : values) result.append(py_grade(value));
    return result;
}
void signals() { if (PyErr_CheckSignals() != 0) throw nb::python_error(); }

/** The mathematical code runs without the GIL. Native polls briefly acquire
 * it for cancellation and optional progress, throttled independently of work.
 */
class CallProgress {
    nb::object callback;
    Clock::time_point start = Clock::now(), last_signal = start, last_report = start;
public:
    explicit CallProgress(nb::object object) : callback(std::move(object)) {
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw nb::type_error("progress must be callable or None");
    }
    Progress function() {
        return [this](const std::string& phase, std::size_t done, std::size_t total, bool force) {
            const auto now = Clock::now();
            const bool check = force || std::chrono::duration<double>(now - last_signal).count() >= .05;
            const bool emit = !callback.is_none() && (force || std::chrono::duration<double>(now - last_report).count() >= .25);
            if (!check && !emit) return;
            nb::gil_scoped_acquire acquire;
            if (check) { signals(); last_signal = now; }
            if (emit) {
                nb::dict event;
                event["phase"] = phase; event["completed"] = done; event["total"] = total;
                event["elapsed_seconds"] = elapsed(start);
                auto response = callback(event);
                if (response.ptr() == Py_False) {
                    PyErr_SetString(PyExc_InterruptedError, "computation cancelled by progress callback");
                    throw nb::python_error();
                }
                last_report = now;
            }
        };
    }
};

std::shared_ptr<Matrix> matrix_input(nb::handle dense, nb::handle sparse, nb::handle rows) {
    if (nb::isinstance<Matrix>(dense)) {
        if (!sparse.is_none() || !rows.is_none()) throw nb::value_error("native matrix cannot be combined with columns or nrows");
        auto result = nb::cast<std::shared_ptr<Matrix>>(dense);
        result->require_available();
        return result;
    }
    IntegerColumns columns;
    if (!sparse.is_none()) {
        if (!dense.is_none() || rows.is_none()) throw nb::value_error("sparse columns require nrows and no dense input");
        const int nrows = index(rows);
        for (auto column : nb::iter(sparse)) {
            MatrixColumn values;
            std::set<int> seen;
            auto items = nb::hasattr(column, "items") ? nb::borrow<nb::object>(column).attr("items")()
                                                   : nb::borrow<nb::object>(column);
            for (auto entry : nb::iter(items)) {
                auto pair = nb::steal<nb::object>(PySequence_Fast(entry.ptr(), "sparse entries must be (row, coefficient) pairs"));
                if (!pair.is_valid()) throw nb::python_error();
                if (PySequence_Fast_GET_SIZE(pair.ptr()) != 2) throw nb::value_error("sparse entries must have two elements");
                int row = index(nb::handle(PySequence_Fast_GET_ITEM(pair.ptr(), 0)));
                if (row >= nrows || !seen.insert(row).second) throw nb::value_error("sparse row outside matrix or repeated");
                auto value = integer(nb::handle(PySequence_Fast_GET_ITEM(pair.ptr(), 1)));
                if (value != 0) values.emplace_back(row, std::move(value));
            }
            std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            columns.push_back(std::move(values));
            signals();
        }
        return std::make_shared<Matrix>(nrows, std::move(columns));
    }
    if (dense.is_none() || !rows.is_none()) throw nb::value_error("provide a dense matrix or columns with nrows");
    int expected_rows = -1;
    const bool shaped = nb::hasattr(dense, "shape");
    if (shaped) {
        auto shape_value = dense.attr("shape");
        auto shape = nb::steal<nb::object>(PySequence_Fast(shape_value.ptr(), "matrix shape must have two dimensions"));
        if (!shape.is_valid()) throw nb::python_error();
        if (PySequence_Fast_GET_SIZE(shape.ptr()) != 2) throw nb::value_error("dense matrix must have two dimensions");
        expected_rows = index(nb::handle(PySequence_Fast_GET_ITEM(shape.ptr(), 0)));
        const int width = index(nb::handle(PySequence_Fast_GET_ITEM(shape.ptr(), 1)));
        if (width > std::numeric_limits<int>::max() - expected_rows)
            throw nb::value_error("augmented matrix exceeds native index range");
        columns.resize(width);
    }
    int row = 0;
    for (auto values : nb::iter(dense)) {
        int col = 0;
        for (auto item : nb::iter(values)) {
            auto value = integer(item);
            if (row == 0 && !shaped) columns.emplace_back();
            if (col >= static_cast<int>(columns.size())) throw nb::value_error("dense matrix rows must have equal length");
            if (value != 0) columns[col].emplace_back(row, std::move(value));
            ++col;
        }
        if (col != static_cast<int>(columns.size())) throw nb::value_error("dense matrix rows must have equal length");
        if (row == std::numeric_limits<int>::max()) throw nb::value_error("matrix exceeds native index range");
        ++row;
        signals();
    }
    if (shaped && row != expected_rows) throw nb::value_error("matrix iteration disagrees with its shape");
    return std::make_shared<Matrix>(row, std::move(columns));
}

PreparedFiltration filtration_input(nb::object input, nb::object times) {
    PreparedFiltration result;
    bool tree = nb::hasattr(input, "get_filtration");
    if (tree && !times.is_none()) throw nb::value_error("SimplexTree already supplies times; regrade when decoding");
    auto iterable = tree ? input.attr("get_filtration")() : input;
    Grades values;
    std::map<mpz_class, int> labels;
    std::set<std::vector<int>> seen;
    for (auto entry : nb::iter(iterable)) {
        nb::object simplex;
        if (tree) {
            auto pair = nb::steal<nb::object>(PySequence_Fast(entry.ptr(), "filtration entries must be (simplex, time) pairs"));
            if (!pair.is_valid()) throw nb::python_error();
            if (PySequence_Fast_GET_SIZE(pair.ptr()) != 2) throw nb::value_error("filtration entries must have two elements");
            simplex = nb::borrow<nb::object>(PySequence_Fast_GET_ITEM(pair.ptr(), 0));
            values.push_back(grade(nb::handle(PySequence_Fast_GET_ITEM(pair.ptr(), 1))));
        } else simplex = nb::borrow<nb::object>(entry);
        std::vector<int> vertices;
        for (auto vertex : nb::iter(simplex)) {
            auto label = integer(vertex);
            auto [it, inserted] = labels.emplace(std::move(label), static_cast<int>(labels.size()));
            vertices.push_back(it->second);
        }
        std::sort(vertices.begin(), vertices.end());
        if (vertices.empty() || std::adjacent_find(vertices.begin(), vertices.end()) != vertices.end())
            throw nb::value_error("simplices must be nonempty and contain distinct vertices");
        if (seen.count(vertices)) throw nb::value_error("duplicate simplex");
        if (vertices.size() > 1) {
            for (std::size_t k = 0; k < vertices.size(); ++k) {
                auto face = vertices; face.erase(face.begin() + k);
                if (!seen.count(face)) throw nb::value_error("each face must occur before its cofaces in the input order");
            }
        }
        seen.insert(vertices); result.simplices.push_back(std::move(vertices));
        if (result.simplices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
            throw nb::value_error("filtration exceeds native index range");
        signals();
    }
    result.vertex_count = static_cast<int>(labels.size());
    if (!times.is_none()) values = grades(times);
    if (tree || !times.is_none()) {
        validate_grades(values, result.simplices.size());
        result.times = std::make_shared<const Grades>(std::move(values));
    }
    return result;
}

const IndexPairs& selected_barcode(const BarcodeResult& result, nb::handle prime) {
    if (prime.is_none()) return result.generic;
    auto p = integer(prime);
    const auto found = result.exceptional.find(p);
    if (found != result.exceptional.end()) return found->second;
    bool valid;
    { nb::gil_scoped_release release; valid = proven_prime(p); }
    if (!valid) throw nb::value_error("characteristic must be a proven prime");
    return result.generic;
}
nb::list barcode(const BarcodeResult& result, nb::handle prime) {
    nb::list output;
    for (const auto& [birth, death] : selected_barcode(result, prime))
        output.append(nb::make_tuple(py_grade((*result.times)[birth]),
            death < 0 ? nb::object(nb::float_(std::numeric_limits<double>::infinity())) : py_grade((*result.times)[death])));
    return output;
}
}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "Native exact arithmetic, all-fields reduction, presentation and barcode algorithms.";
    nb::class_<Matrix>(m, "Matrix")
        .def_prop_ro("shape", [](const Matrix& a) { return nb::make_tuple(a.rows, a.cols); })
        .def_ro("consumed", &Matrix::consumed)
        .def_prop_ro("identity", [](const Matrix& a) { return a.identity; })
        .def("copy", &Matrix::copy, "Copy unconsumed native coefficients explicitly before starting reduction.");
    nb::class_<MatrixData>(m, "MatrixData")
        .def_ro("matrix", &MatrixData::matrix)
        .def_ro("module", &MatrixData::module).def_ro("degree", &MatrixData::degree)
        .def_ro("source_count", &MatrixData::source_count)
        .def_prop_ro("dimensions", [](const MatrixData& data) { return data.field ? data.field->dimensions : std::vector<int>{}; })
        .def_prop_ro("simplex_sources", [](const MatrixData& data) { return data.field ? data.field->simplex_sources : std::vector<int>{}; })
        .def_ro("birth_sources", &MatrixData::birth_sources).def_ro("relation_sources", &MatrixData::relation_sources)
        .def_ro("timings", &MatrixData::timings)
        .def_prop_ro("times", [](const MatrixData& data) -> nb::object {
            return data.times ? nb::object(py_grades(*data.times)) : nb::object(nb::none());
        });
    nb::class_<ReductionState>(m, "ReductionState")
        .def_prop_ro("phase", [](const ReductionState& s) { return phase_name(s.phase); })
        .def_prop_ro("shape", [](const ReductionState& s) { return nb::make_tuple(s.rows, s.cols); })
        .def_ro("generic_pivots", &ReductionState::generic_pivots)
        .def_prop_ro("primes", [](const ReductionState& s) { return py_integers(s.primes); })
        .def_prop_ro("exceptional_pivots", [](const ReductionState& s) {
            s.require(Phase::complete); nb::dict result;
            for (const auto& [p, profile] : s.exceptional_pivots) result[py_integer(p)] = nb::cast(profile);
            return result;
        })
        .def_ro("timings", &ReductionState::timings)
        .def("pivot_profile", [](const ReductionState& s, nb::object prime) {
            s.require(Phase::complete);
            if (prime.is_none()) return s.generic_pivots;
            const auto p = integer(prime);
            auto found = s.exceptional_pivots.find(p);
            if (found != s.exceptional_pivots.end()) return found->second;
            bool valid;
            { nb::gil_scoped_release release; valid = proven_prime(p); }
            if (!valid) throw nb::value_error("characteristic must be a proven prime");
            return s.generic_pivots;
        }, "prime"_a = nb::none(), "Return the augmented canonical pivot profile in characteristic zero or prime p.");
    nb::class_<BarcodeResult>(m, "BarcodeResult")
        .def_ro("module", &BarcodeResult::module).def_ro("degree", &BarcodeResult::degree)
        .def_ro("timings", &BarcodeResult::timings)
        .def_prop_ro("matrix_primes", [](const BarcodeResult& b) { return py_integers(b.matrix_primes); })
        .def_prop_ro("exceptional_primes", [](const BarcodeResult& b) {
            nb::list result; for (const auto& [p, bars] : b.exceptional) result.append(py_integer(p)); return result;
        })
        .def_prop_ro("field_independent", [](const BarcodeResult& b) { return b.exceptional.empty(); })
        .def("barcode", &barcode, "prime"_a = nb::none(),
             "List (birth, death) pairs, preserving multiplicity; None selects characteristic zero.")
        .def("index_pairs", [](const BarcodeResult& b, nb::object p) { return selected_barcode(b, p); },
             "prime"_a = nb::none(), "Source-index pairs after removing zero-length intervals; -1 means infinite death.");
    m.def("rational_reduction", [](nb::object a, nb::object columns, nb::object nrows, nb::object callback) {
        CallProgress progress(std::move(callback));
        auto matrix = matrix_input(a, columns, nrows);
        nb::gil_scoped_release release;
        return rational_reduction(*matrix, progress.function());
    }, "matrix"_a = nb::none(), "columns"_a = nb::none(), "nrows"_a = nb::none(), "progress"_a = nb::none());
    m.def("exhaustive_rational_reduction", [](ReductionState& s, nb::object callback) {
        CallProgress progress(std::move(callback)); nb::gil_scoped_release release;
        exhaustive_rational_reduction(s, progress.function());
    }, "state"_a, "progress"_a = nb::none());
    m.def("exceptional_correction", [](ReductionState& s, nb::object callback) {
        CallProgress progress(std::move(callback)); nb::gil_scoped_release release;
        exceptional_correction(s, progress.function());
    }, "state"_a, "progress"_a = nb::none());
    m.def("build_matrices", [](nb::object simplices, nb::object times, nb::object degrees,
                                const std::vector<std::string>& modules, nb::object callback) {
        CallProgress progress(std::move(callback));
        std::vector<int> requested;
        for (auto degree : nb::iter(degrees)) requested.push_back(index(degree));
        auto input = filtration_input(std::move(simplices), std::move(times));
        nb::gil_scoped_release release;
        return build_matrices(input, requested, modules, progress.function());
    }, "simplices"_a, "times"_a.none(), "degrees"_a, "modules"_a, "progress"_a = nb::none());
    m.def("decode_barcodes", [](const ReductionState& state, const MatrixData& data, nb::object input) {
        std::shared_ptr<const Grades> values;
        if (!input.is_none()) values = std::make_shared<const Grades>(grades(input));
        else if (data.times) values = data.times;
        else {
            Grades indices;
            for (std::size_t i = 0; i < data.source_count; ++i) indices.emplace_back(mpz_class(i));
            values = std::make_shared<const Grades>(std::move(indices));
        }
        nb::gil_scoped_release release;
        return decode_barcodes(state, data, std::move(values));
    }, "state"_a, "data"_a, "times"_a = nb::none());
    m.def("ordinary_pairs_timed", &ordinary_pairs_timed, "sequences"_a, "dimensions"_a, "degree"_a,
          nb::call_guard<nb::gil_scoped_release>());
    m.def("_is_prime", [](nb::object p) { auto value = integer(p); nb::gil_scoped_release release; return proven_prime(value); });
}
