from collections.abc import Mapping
import math
from numbers import Integral, Real


_METHODS = ("modular", "ours")
_CANONICAL_METHODS = ("brute_force", "modular", "ours")
_ALL_METHODS = ("brute_force", "modular", "omnifield", "ours")
_PHASES = ("prime_discovery", "method_initialization", "main_computation", "output")


def _integer(value, name, *, minimum=0):
    """Validate integer counts without silently truncating fractional values."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _seconds(value, name, *, positive=False):
    """Validate a measured duration or a configured positive timeout."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return float(value)


def _prime_list(value):
    """Normalize an already-recorded candidate list without asserting primality."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError("A prime list must contain distinct integers >= 2.")
    try:
        candidates = tuple(_integer(p, "Prime candidate", minimum=2) for p in value)
    except TypeError as error:
        raise ValueError("A prime list must contain distinct integers >= 2.") from error
    if len(set(candidates)) != len(candidates):
        raise ValueError("A prime list must not contain duplicates.")
    return tuple(sorted(candidates))


def _case_counts(record, trials):
    """Recover completed and failed trial counts with legacy timeout support."""
    status = record.get("status")
    if status not in ("success", "partial", "timeout", "memory_limit", "failed", "error"):
        raise ValueError(f"Unsupported result status: {status!r}.")
    total = _integer(record.get("trials", trials), "trials", minimum=1)
    completed = _integer(record.get("successful_trials", total if status == "success" else 0),
                         "successful_trials")
    memory = _integer(record.get("memory_limited_trials", total if status == "memory_limit" else 0),
                      "memory_limited_trials")
    errors = _integer(record.get("failed_trials", total if status in ("failed", "error") else 0),
                      "failed_trials")
    timeouts = _integer(record.get("timed_out_trials", total - completed - memory - errors),
                        "timed_out_trials")
    if completed + timeouts + memory + errors != total:
        raise ValueError("Inconsistent trial counts.")
    expected = ("success" if completed == total else "partial" if completed else
                "timeout" if timeouts == total else "memory_limit" if memory == total
                else "failed")
    if status != expected and not (status == "error" and errors == total):
        raise ValueError("Trial counts disagree with result status.")
    phases = record.get("phases_seconds")
    if not isinstance(phases, Mapping) or any(phase not in phases for phase in _PHASES):
        raise ValueError("Every result must contain all four measured phases.")
    if completed:
        duration = math.fsum(_seconds(phases[phase], phase) for phase in _PHASES)
    else:
        if any(phases[phase] is not None for phase in _PHASES):
            raise ValueError("A result without completed trials must have unknown phase times.")
        duration = None
    return total, completed, (timeouts, memory, errors), duration


def vertex_comparison_data(RESULTS_ALL, vertices, *, trials=1):
    """Aggregate recorded modular/ours results by the supplied vertex counts.

    ``RESULTS_ALL[j]`` maps method names to one summary per filtration for
    ``vertices[j]``. Vertex counts must be distinct positive integers; outputs
    are sorted by this count. A missing method has zero attempts and an unknown
    mean, rather than invented failed trials. Method arrays follow
    ``("modular", "ours")`` and failure channels are timeout, memory, error.

    Runtime is the sum of the four recorded phases averaged over completed
    trials, weighting each case mean by its successful-trial count. Limits and
    failed trials do not contribute runtimes. Prime products are averaged
    arithmetically once per filtration with known canonical candidates, without
    trial weights or duplication across methods. Empty lists have product one;
    missing lists are excluded and reported in ``known_prime_cases``. Products
    and their sum are exact Python integers. ``prime_product_log10`` contains
    the logarithm of their arithmetic mean, avoiding floating-point overflow;
    it is not the mean of their logarithms. A missing mean is ``math.nan``.

    Canonical lists from screening, brute force, modular and ours must agree
    where available. Omnifield's own list is not a canonical reference. These
    are recorded candidates, not a new primality or all-fields certificate.
    Repeated cases count separately even if they have identical input digests.
    No supplied object is modified.
    """
    trials = _integer(trials, "trials", minimum=1)
    results = tuple(RESULTS_ALL)
    vertices = tuple(_integer(n, "Vertex count", minimum=1) for n in vertices)
    if not results or len(results) != len(vertices):
        raise ValueError("RESULTS_ALL and vertices must have the same nonzero length.")
    if len(set(vertices)) != len(vertices):
        raise ValueError("Vertex counts must be distinct.")
    ordered = sorted(zip(vertices, results), key=lambda pair: pair[0])
    output = {
        "vertices": tuple(n for n, _ in ordered),
        "methods": _METHODS,
        "mean_seconds": [[math.nan] * len(ordered) for _ in _METHODS],
        "attempted": [[0] * len(ordered) for _ in _METHODS],
        "completed": [[0] * len(ordered) for _ in _METHODS],
        "failures": [[[0, 0, 0] for _ in ordered] for _ in _METHODS],
        "prime_product_log10": [],
        "known_prime_cases": [],
        "case_counts": [],
    }
    timeout_limits = set()
    trial_counts = set()
    for j, (n, batch) in enumerate(ordered):
        if not isinstance(batch, Mapping):
            raise ValueError("Each vertex-count result must be a method dictionary.")
        records = {}
        for method in _ALL_METHODS:
            if method not in batch:
                continue
            value = batch[method]
            if isinstance(value, (str, bytes, Mapping)):
                raise ValueError(f"{method} results must be a sequence of case records.")
            try:
                records[method] = tuple(value)
            except TypeError as error:
                raise ValueError(f"{method} results must be a sequence of case records.") from error
            if any(not isinstance(record, Mapping) for record in records[method]):
                raise ValueError("Each case record must be a dictionary.")
        lengths = {len(values) for values in records.values() if values}
        if not any(records.get(method) for method in _METHODS):
            raise ValueError(f"No modular or ours results are available for n={n}.")
        if len(lengths) != 1:
            raise ValueError("All available methods must have the same number of cases.")
        cases = lengths.pop()
        output["case_counts"].append(cases)
        for m, method in enumerate(_METHODS):
            weighted = []
            for record in records.get(method, ()):
                if "method" in record and record["method"] != method:
                    raise ValueError("Record method disagrees with its method dictionary key.")
                total, completed, failures, duration = _case_counts(record, trials)
                trial_counts.add(total)
                output["attempted"][m][j] += total
                output["completed"][m][j] += completed
                for channel, number in enumerate(failures):
                    output["failures"][m][j][channel] += number
                if completed:
                    weighted.append((duration, completed))
                if "timeout_seconds" in record:
                    timeout_limits.add(_seconds(record["timeout_seconds"], "timeout_seconds", positive=True))
                for limit in record.get("timeout_limits_seconds", ()):
                    timeout_limits.add(_seconds(limit, "timeout_limits_seconds", positive=True))
            completed = output["completed"][m][j]
            if completed:
                output["mean_seconds"][m][j] = math.fsum(
                    duration * (weight / completed) for duration, weight in weighted
                )
        product_sum = 0
        known = 0
        for i in range(cases):
            case_records = {method: values[i] for method, values in records.items() if values}
            for key in ("seed", "input_sha256"):
                values = [record[key] for record in case_records.values()
                          if record.get(key) is not None]
                if values and any(value != values[0] for value in values[1:]):
                    raise ValueError(f"Case {i} at n={n}: methods disagree on {key}.")
            references = []
            for method, record in case_records.items():
                keys = ("reference_primes", "primes") if method in _CANONICAL_METHODS else ("reference_primes",)
                for key in keys:
                    candidates = _prime_list(record.get(key))
                    if candidates is not None:
                        references.append(candidates)
            if references:
                if any(value != references[0] for value in references[1:]):
                    raise ValueError(f"Case {i} at n={n}: prime discovery disagrees between methods.")
                product_sum += math.prod(references[0])
                known += 1
        output["known_prime_cases"].append(known)
        output["prime_product_log10"].append(
            math.log10(product_sum) - math.log10(known) if known else math.nan
        )
    output["timeout_limits_seconds"] = tuple(sorted(timeout_limits))
    output["trials_per_case"] = tuple(sorted(trial_counts))
    return output


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

def _comparison_plot_data(RESULTS, trials):
    """Validate cases and recover completion weights, failure counts and primes.

    Screening or completed canonical-discovery lists determine groups.
    Omnifield's additional candidates cannot supply a canonical count.
    Missing timings have zero weight, never an imputed runtime.
    The final array counts timeouts, memory limits and other errors separately.
    Older success/partial/timeout records may omit the latter two counts.
    """
    methods = ["brute_force", "modular", "omnifield", "ours"]
    phases = ["prime_discovery", "method_initialization", "main_computation", "output"]
    cases = len(RESULTS["brute_force"])
    if cases == 0 or any(len(RESULTS[m]) != cases for m in methods):
        raise ValueError("All methods must have the same nonzero number of cases.")

    completed = np.zeros((len(methods), cases), dtype=int)
    attempted = np.zeros_like(completed)
    failures = np.zeros((len(methods), cases, 3), dtype=int)
    # Zero is only an internal placeholder with weight zero, never a measured time.
    times = np.zeros((len(methods), cases, len(phases)), dtype=float)
    for m, method in enumerate(methods):
        for i, record in enumerate(RESULTS[method]):
            status = record["status"]
            if status not in ("success", "partial", "timeout", "memory_limit", "failed", "error"):
                raise ValueError(f"Unsupported result status: {status}")
            total = int(record.get("trials", trials))
            successes = int(record.get("successful_trials", total if status == "success" else 0))
            memory = int(record.get("memory_limited_trials", total if status == "memory_limit" else 0))
            errors = int(record.get("failed_trials", total if status in ("failed", "error") else 0))
            timeouts = int(record.get("timed_out_trials", total - successes - memory - errors))
            counts = (successes, timeouts, memory, errors)
            if total <= 0 or min(counts) < 0 or sum(counts) != total:
                raise ValueError("Inconsistent trial counts.")
            expected = ("success" if successes == total else "partial" if successes else
                        "timeout" if timeouts == total else "memory_limit" if memory == total
                        else "failed")
            if status != expected and not (status == "error" and errors == total):
                raise ValueError("Trial counts disagree with result status.")
            attempted[m, i], completed[m, i] = total, successes
            failures[m, i] = timeouts, memory, errors
            if successes:
                values = np.asarray([record["phases_seconds"][p] for p in phases], dtype=float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Completed phase times must be finite and nonnegative.")
                times[m, i] = values
            elif any(record["phases_seconds"][p] is not None for p in phases):
                raise ValueError("A result without completed trials must have unknown phase times.")

    prime_counts = []
    for i in range(cases):
        screening = [set(map(int, RESULTS[m][i]["reference_primes"]))
                     for m in methods if RESULTS[m][i].get("reference_primes") is not None]
        # Retain compatibility with earlier successful-only RESULTS dictionaries.
        known = [set(map(int, RESULTS[m][i]["primes"]))
                 for m in ("brute_force", "modular", "ours")
                 if RESULTS[m][i].get("primes") is not None]
        references = screening + known
        if not references:
            raise ValueError(f"No reference prime list is available for case {i}; canonical discovery did not complete.")
        if any(primes != references[0] for primes in references[1:]):
            raise ValueError(f"Prime discovery disagrees for case {i}.")
        prime_counts.append(len(references[0]))
    prime_counts = np.asarray(prime_counts)
    return methods, times, completed, attempted, prime_counts, failures

def plot_comparison_results(RESULTS, *, n, count, trials):
    """Plot completed-trial phase means and explicitly mark censored trials.

    Phase means are weighted by each case's successful-trial count. Failed
    trials have no measured phase breakdown; time and memory limits appear in
    separate failure annotations, never bar segments. Screening primes keep failed cases in
    the appropriate panel without claiming that the failed method found them.

    In each panel, Omnifield is truncated at twice the largest other-method
    total if it exceeds that threshold. Its full mean remains in the label;
    phase segments retain their original scale. Without a positive completed
    reference total, no truncation is applied. Grouped panels share an axis
    range computed from the displayed totals.
    """
    methods = ["brute_force", "modular", "omnifield", "ours"]
    method_labels = ["Brute force", "Modular\nreconstruction", "Omnifield", "Our method"]
    phase_labels = ["Prime discovery", "Initialization", "Main phase", "Barcode output"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    methods, times, completed, attempted, prime_counts, failures = _comparison_plot_data(RESULTS, trials)
    cases = times.shape[1]
    has_failures = bool(np.any(failures))

    def aggregate(indices):
        weights = completed[:, indices]
        successes = weights.sum(axis=1)
        attempts = attempted[:, indices].sum(axis=1)
        sums = (times[:, indices, :] * weights[:, :, None]).sum(axis=1)
        means = np.divide(sums, successes[:, None], out=np.zeros_like(sums),
                          where=successes[:, None] != 0)
        return means, successes, attempts, failures[:, indices, :].sum(axis=1)

    def displayed_totals(summary):
        totals = summary[0].sum(axis=1)
        omni = methods.index("omnifield")
        reference = max(total for i, total in enumerate(totals) if i != omni)
        if reference > 0:
            totals[omni] = min(totals[omni], 2 * reference)
        return totals

    def limits_label(method, indices, kind):
        limits = set()
        value_key, values_key = (
            ("timeout_seconds", "timeout_limits_seconds")
            if kind == "timeout" else
            ("memory_limit_mb", "memory_limits_mb")
        )
        for i in indices:
            record = RESULTS[method][i]
            failure_index = 0 if kind == "timeout" else 1
            if failures[methods.index(method), i, failure_index]:
                if record.get(value_key) is not None:
                    limits.add(float(record[value_key]))
                limits.update(float(value) for value in record.get(values_key, ())
                              if value is not None)
        if not limits:
            return "worker limit unspecified" if kind == "timeout" else None
        values = ", ".join(f"{value:g}" for value in sorted(limits))
        return (f"{values} s limit{'s' if len(limits) > 1 else ''}" if kind == "timeout"
                else f"{values} MiB")

    def failure_label(method, indices, counts, attempts):
        labels = []
        timeouts, memory, errors = counts
        if timeouts:
            labels.append(f"{timeouts}/{attempts} timed out ({limits_label(method, indices, 'timeout')})")
        if memory:
            limit = limits_label(method, indices, "memory")
            label = f"memory limit ({limit})" if limit else "memory allocation failed"
            labels.append(f"{memory}/{attempts} {label}")
        if errors:
            labels.append(f"{errors}/{attempts} error{'s' if errors != 1 else ''}")
        return " · ".join(labels)

    def draw(ax, summary, indices, title, xmax):
        means, successes, attempts, failure_counts = summary
        totals = means.sum(axis=1)
        displayed = displayed_totals(summary)
        positions = np.arange(len(methods))
        valid = successes > 0
        left = np.zeros(len(methods))
        bar_height = 0.62
        bar_patches = {y: [] for y in positions[valid]}
        for j, color in enumerate(colors):
            widths = np.minimum(means[:, j], np.maximum(displayed - left, 0))
            bars = ax.barh(positions[valid], widths[valid], left=left[valid],
                           height=bar_height, color=color)
            for y, patch in zip(positions[valid], bars):
                bar_patches[y].append(patch)
            left += widths
        for y, method in enumerate(methods):
            if valid[y]:
                label = f"{totals[y]:.3g} s"
                if displayed[y] < totals[y]:
                    label += "\n(truncated)"
                    # Cut a full-height zigzag into the stack's right edge.
                    # Clipping preserves the background grid and all phase colors.
                    cut_width = min(0.012 * xmax, 0.15 * left[y])
                    edge_y = np.linspace(y - bar_height / 2, y + bar_height / 2, 9)
                    edge_x = left[y] - cut_width * (np.arange(len(edge_y)) % 2)
                    outline = [(0, edge_y[0]), *zip(edge_x, edge_y), (0, edge_y[-1])]
                    clip = plt.Polygon(outline, closed=True, transform=ax.transData)
                    for patch in bar_patches[y]:
                        patch.set_clip_path(clip)
                    ax.plot(edge_x, edge_y, color="#555555", linewidth=0.8,
                            solid_joinstyle="miter", zorder=3)
                ax.annotate(label, (left[y], y), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=9)
            else:
                ax.text(0.02, y, "No completed trials", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=9, color="#a12622")
            missing = attempts[y] - successes[y]
            if missing:
                ax.plot(left[y], y, marker="x", color="#a12622", markersize=7,
                        markeredgewidth=1.5, clip_on=False)
                ax.text(0.98, y + bar_height / 2 + 0.04,
                        failure_label(method, indices, failure_counts[y], attempts[y]),
                        transform=ax.get_yaxis_transform(), ha="right", va="top",
                        fontsize=8, color="#a12622", clip_on=True)
        ax.set_yticks(positions, method_labels)
        ax.set_ylim(len(methods) - 0.4, -0.5)
        ax.set_xlim(0, xmax or 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.set_xlabel("Mean phase time of completed trials (seconds)" if has_failures
                      else "Mean phase time (seconds)")
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, phase_labels)]
    parameters = (f"{n} vertices, {count} triangles, {cases} filtrations, {trials} trials per case")

    def finish(fig, title):
        height = fig.get_figheight()
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.1 / height)
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1 - 0.40 / height),
                   ncol=4, frameon=False, fontsize=10)
        # if has_failures:
        #     fig.text(0.5, 0.015,
        #              "× marks failed trials. Bars average completed trials only; "
        #              "limits include worker startup. No phase time is imputed.",
        #              ha="center", fontsize=8, color="#555555")
        # tight_layout reserves the title; add room below it for the legend.
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / height), h_pad=2.0)

    indices = np.arange(cases)
    overall = aggregate(indices)
    fig_all, ax = plt.subplots(figsize=(9.8, 4.9), facecolor="white")
    draw(ax, overall, indices, "", displayed_totals(overall).max() * 1.10)
    # Keep four round ticks inside the tight axis range, rather than putting
    # a tick at its arbitrary data-dependent endpoint.
    target_step = ax.get_xlim()[1] / 3
    magnitude = 10.0 ** np.floor(np.log10(target_step))
    steps = magnitude * np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 10])
    tick_step = steps[steps <= target_step][-1]
    ax.set_xticks(tick_step * np.arange(4))
    finish(fig_all, f"Running times — {parameters}")

    groups = np.unique(prime_counts)
    grouped_indices = [np.flatnonzero(prime_counts == k) for k in groups]
    grouped = [aggregate(indices) for indices in grouped_indices]
    xmax = max(displayed_totals(summary).max() for summary in grouped) * 1.25
    rows = (len(groups) + 2) // 3
    columns = (len(groups) + rows - 1) // rows
    fig_groups, axes = plt.subplots(
        rows, columns, squeeze=False, sharex=True,
        figsize=(max(9.8, 5.3 * columns), 3.2 * rows + 1.3), facecolor="white",
    )
    for ax, k, summary, indices in zip(axes.flat, groups, grouped, grouped_indices):
        title = (f"{k} exceptional prime{'s' if k != 1 else ''}"
                 f" ({len(indices)} case{'s' if len(indices) != 1 else ''})")
        draw(ax, summary, indices, title, xmax)
        ax.tick_params(axis="x", labelbottom=True)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    finish(fig_groups, f"Running times grouped by number of exceptional primes — {parameters}")
    return fig_all, fig_groups

def plot_comparison_curves(RESULTS, *, n, count, trials, log_scale=True,
                           time_unit="ms", height=4.9, figsize=None):
    """Plot mean summed phase time against the number of exceptional primes.

    Sum the four measured phases of each completed trial, then average within
    each prime-count group. Case means are weighted by successful trial counts,
    just as in ``plot_comparison_results``. These are measured phase totals,
    not worker wall times. No smoothing or fitted curves are used.

    The lower strip counts filtrations, including cases whose trials failed.
    If any trials failed, a separate strip reports timeout, memory-limit and
    other-error counts per method and group. Groups without completions have
    gaps in their curve; limits are never plotted as runtimes. Partial means are conditional
    on completion and can cover different case mixes for different methods.

    ``time_unit`` is ``"ms"`` or ``"s"``. The default logarithmic y-axis
    requires positive completed means; use ``log_scale=False`` to show zeros.
    ``height`` is the figure height in inches; its default matches the overall
    bar plot. An explicit ``figsize=(width, height)`` overrides this parameter.
    Return the Matplotlib Figure for display, customization or saving.
    """
    from matplotlib.ticker import LogLocator, NullFormatter

    if time_unit not in ("ms", "s"):
        raise ValueError('time_unit must be "ms" or "s".')
    factor = 1000.0 if time_unit == "ms" else 1.0
    methods, times, completed, attempted, prime_counts, failures = _comparison_plot_data(RESULTS, trials)
    labels = ["Brute force", "Modular reconstruction", "Omnifield", "Our method"]
    # colors = ["#0072B2", "#E69F00", "#CC79A7", "#222222"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#222222"]
    markers = ["s", "^", "D", "o"]
    groups = np.unique(prime_counts)
    grid = np.arange(groups[0], groups[-1] + 1)
    case_counts = np.array([np.count_nonzero(prime_counts == k) for k in grid])
    means = np.full((len(methods), len(groups)), np.nan)
    successes = np.zeros_like(means, dtype=int)
    attempts = np.zeros_like(successes)
    failure_counts = np.zeros((len(methods), len(groups), 3), dtype=int)
    totals = times.sum(axis=2)
    for j, k in enumerate(groups):
        indices = np.flatnonzero(prime_counts == k)
        weights = completed[:, indices]
        successes[:, j] = weights.sum(axis=1)
        attempts[:, j] = attempted[:, indices].sum(axis=1)
        failure_counts[:, j] = failures[:, indices, :].sum(axis=1)
        weighted = (totals[:, indices] * weights).sum(axis=1)
        np.divide(weighted * factor, successes[:, j], out=means[:, j],
                  where=successes[:, j] > 0)
    finite = means[np.isfinite(means)]
    if log_scale and np.any(finite <= 0):
        raise ValueError("Log scale requires positive completed means; use log_scale=False.")
    missed = attempts - successes
    has_failures = bool(np.any(missed))

    # Keep the reference's main curves and shared-x case-count strip. Censored
    # counts get their own strip so their vertical position cannot imply time.
    rows = 3 if has_failures else 2
    ratios = [4.5, 1.1, 0.9] if has_failures else [4.5, 0.9]
    if figsize is None:
        figsize = (12, height)
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=figsize,
                             gridspec_kw={"height_ratios": ratios, "hspace": 0.12},
                             facecolor="white")
    figure_height = fig.get_figheight()
    ax, cases_ax = axes[0], axes[-1]
    for m, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        ax.plot(groups, means[m], label=label, color=color, marker=marker,
                linewidth=2.8 if methods[m] == "ours" else 1.8,
                markersize=6.5, markeredgewidth=1.5,
                markerfacecolor=color)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim((finite.min() / 1.3, finite.max() * 1.3) if finite.size else (1, 10))
    else:
        ax.set_ylim(0, finite.max() * 1.15 if finite.size and finite.max() > 0 else 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_ylabel(f"Mean summed phase time\n({time_unit}{', log scale' if log_scale else ''})",
                  fontsize=12)
    ax.grid(axis="y", color="#d9dee5", linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if not finite.size:
        ax.text(0.5, 0.5, "No completed trials", transform=ax.transAxes,
                ha="center", va="center", color="#a12622", fontsize=13)

    if has_failures:
        failure_ax = axes[1]
        short_labels = ["Brute force", "Modular", "Omnifield", "Ours"]
        failure_ax.set_yticks(range(len(methods)), short_labels)
        failure_ax.set_ylim(len(methods) - 0.5, -0.5)
        failure_label = "Not\ncompleted" if np.any(failure_counts[:, :, 1:]) else "Timed out"
        failure_ax.set_ylabel(failure_label, rotation=90, ha="center", va="bottom",
                              fontsize=11, labelpad=8)
        failure_ax.tick_params(axis="both", length=0, labelbottom=False, labelsize=9)
#        failure_ax.text(1, 1.04, "T: timeout · M: memory limit · E: error",
#                        transform=failure_ax.transAxes, ha="right", va="bottom",
#                        fontsize=7, color="#555555")
        for m in range(len(methods)):
            for j, k in enumerate(groups):
                total_failures = int(failure_counts[m, j].sum())
                text = f"{total_failures}/{attempts[m, j]}"
#                counts = failure_counts[m, j]
#                text = " · ".join(f"{kind} {number}/{attempts[m, j]}"
#                                  for kind, number in zip(("T", "M", "E"), counts)
#                                  if number) or "–"
                failure_ax.text(k, m, text, ha="center", va="center", fontsize=8,
                                color=colors[m] if missed[m, j] else "#aeb5be")
        for spine in failure_ax.spines.values():
            spine.set_visible(False)

    cases_ax.bar(grid, case_counts, width=0.64, color="#bec6d1", zorder=2)
    for k, number in zip(grid, case_counts):
        cases_ax.annotate(str(number), (k, number), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=10,
                          color="#414b57")
    cases_ax.set_ylim(0, max(case_counts) * 1.4)
    cases_ax.set_yticks([])
    cases_ax.set_ylabel("Cases", rotation=0, ha="right", va="center",
                        fontsize=12, color="#414b57")
    # Keep this label close to its panel instead of aligning it with the others.
    cases_ax.yaxis.set_label_coords(-0.01, 0.5)
    # Every integer is visible for ordinary ranges; wide ranges avoid crowding.
    if len(grid) <= 35:
        cases_ax.set_xticks(grid)
    else:
        cases_ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    cases_ax.set_xlabel("Number of exceptional primes", fontsize=13, labelpad=10)
    cases_ax.set_xlim(grid[0] - 0.6, grid[-1] + 0.6)
    cases_ax.spines["left"].set_visible(False)
    for axis in (ax, cases_ax):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#6b7280")
        axis.tick_params(labelsize=10)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")

    fig.suptitle(f"Running times grouped by primes — {n} vertices, {count} triangles, {len(prime_counts)} filtrations, {trials} trials per case",
                 fontsize=13, fontweight="bold", y=1 - 0.10 / figure_height)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.40 / figure_height),
               ncol=4, frameon=False, fontsize=12)
    # if has_failures:
    #     fig.text(0.5, 0.10 / figure_height,
    #              "Means use completed trials only; failures are counted separately. "
    #              "Case sets may differ between methods.",
    #              ha="center", fontsize=9, color="#555555")
    # Reserve margins in inches so text stays separated as height changes.
    fig.subplots_adjust(left=0.14 if has_failures else 0.13, right=0.98, top=1 - 0.80 / figure_height,
                        bottom=(0.95 if has_failures else 0.65) / figure_height)
    fig.align_ylabels(axes)
    if has_failures:
        # Align the first lines, rather than the units line of a multiline title.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_axes = (ax, failure_ax)
        left_edges = [axis.yaxis.label.get_window_extent(renderer).x0 for axis in label_axes]
        for axis, left in zip(label_axes, left_edges):
            axis.yaxis.labelpad += (left - min(left_edges)) * 72 / fig.dpi
    return fig

def plot_comparison_by_vertices(RESULTS_ALL, *, vertices, trials=1, c=None,
                               log_scale=True, time_unit="s", height=7.0,
                               figsize=None):
    """Compare modular reconstruction and ours over a sweep of vertex counts.

    ``vertices`` gives one n for each entry of RESULTS_ALL, in the same order.
    The top panel averages summed phase times over completed trials, weighted
    by successful-trial counts. The middle panel records timeouts separately
    from memory failures and errors; an absent method is marked "not run".

    The bottom panel is the arithmetic mean of products of exceptional primes,
    counting each filtration once across methods and trials. Only available
    canonical prime lists contribute; coverage counts are not drawn.
    An empty prime list has product one, while an unknown list remains missing.
    Exact integer products are averaged before taking log10, avoiding floating
    overflow even for products with thousands of digits. The bottom axis uses
    log10 coordinates with product-valued ticks; it is not a geometric mean.
    Recorded candidates are plotted without upgrading their primality status.

    Return a Figure with three shared-x panels. Missing measurements leave
    gaps; time limits are never used as measured runtimes. ``height`` is in
    inches; ``figsize=(width, height)`` overrides it. ``trials`` supplies the
    count only for older records that omit it. No benchmarks are executed.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, StrMethodFormatter

    if time_unit not in ("ms", "s"):
        raise ValueError('time_unit must be "ms" or "s".')
    if c is not None and (not math.isfinite(c) or c <= 0):
        raise ValueError("c must be positive and finite.")
    data = vertex_comparison_data(RESULTS_ALL, vertices, trials=trials)
    x = np.asarray(data["vertices"])
    means = np.asarray(data["mean_seconds"]) * (1000 if time_unit == "ms" else 1)
    attempted = np.asarray(data["attempted"])
    failures = np.asarray(data["failures"])
    products = np.asarray(data["prime_product_log10"])
    finite = means[np.isfinite(means)]
    if log_scale and np.any(finite <= 0):
        raise ValueError("Log scale requires positive completed means; use log_scale=False.")

    if figsize is None:
        figsize = (12, height)
    extra_failures = bool(np.any(failures[:, :, 1:]))
    fig, (ax, failure_ax, product_ax) = plt.subplots(
        3, 1, sharex=True, figsize=figsize, facecolor="white",
        gridspec_kw={"height_ratios": [4, 1.6 if extra_failures else 1.0, 1.8],
                     "hspace": 0.25},
    )
    labels = ["Modular reconstruction", "Our method"]
    colors = ["#ff7f0e", "#222222"]
    for m, (label, color, marker) in enumerate(zip(labels, colors, ("^", "o"))):
        ax.plot(x, means[m], label=label, color=color, marker=marker,
                markerfacecolor=color, markersize=6.5, markeredgewidth=1.2,
                linewidth=2.8 if data["methods"][m] == "ours" else 1.8)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim((finite.min() / 1.3, finite.max() * 1.3) if finite.size else (1, 10))
    else:
        ax.set_ylim(0, finite.max() * 1.15 if finite.size and finite.max() > 0 else 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_ylabel(f"Mean running time \n({time_unit}{', log scale' if log_scale else ''})",
                  fontsize=12)
    if not finite.size:
        ax.text(0.5, 0.5, "No completed trials", transform=ax.transAxes,
                ha="center", va="center", color="#a12622", fontsize=12)

    failure_ax.set_yticks([0, 1], ["Modular reconstruction", "Ours"])
    failure_ax.set_ylim(1.6, -0.6)
    failure_label = "Not completed" if extra_failures else "Timed out"
    limits = data["timeout_limits_seconds"]
    if limits:
        failure_label += "\n(limit: " + ", ".join(f"{value:g}" for value in limits) + " s)"
    failure_ax.set_ylabel(failure_label,
                          rotation=90, ha="center", va="bottom", fontsize=11, labelpad=9)
    failure_ax.tick_params(axis="both", length=0, labelbottom=False, labelsize=9)
    if extra_failures:
        caption = "T: timeout · M: memory limit · E: error"
        failure_ax.text(1, 1.02, caption, transform=failure_ax.transAxes,
                        ha="right", va="bottom", fontsize=8, color="#555555")
    for m in range(2):
        for j, n in enumerate(x):
            total = attempted[m, j]
            timed_out, memory, errors = failures[m, j]
            if total == 0:
                text = "not run"
            elif extra_failures:
                text = f"T {timed_out}/{total}"
                text += "".join(f"\n{label} {number}/{total}"
                                for label, number in (("M", memory), ("E", errors)) if number)
            else:
                text = f"{timed_out}/{total}"
            failure_ax.text(n, m, text, ha="center", va="center", fontsize=11,
                            color=colors[m] if failures[m, j].sum() else "#8a939f")
    for spine in failure_ax.spines.values():
        spine.set_visible(False)

    product_ax.plot(x, products, color="black", marker="o", markersize=4.5,
                    markerfacecolor="black", linewidth=1.8)
    valid_products = products[np.isfinite(products)]
    if valid_products.size:
        pad = max(0.2, 0.18 * np.ptp(valid_products))
        product_ax.set_ylim(max(0, valid_products.min() - pad), valid_products.max() + pad)
    else:
        product_ax.set_ylim(0, 1)
        product_ax.text(0.5, 0.55, "No prime lists available", transform=product_ax.transAxes,
                        ha="center", va="center", color="#a12622", fontsize=10)
    product_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, min_n_ticks=2))
    product_ax.yaxis.set_major_formatter(FuncFormatter(lambda exponent, _: rf"$10^{{{exponent:g}}}$"))
    product_ax.set_ylabel("Mean prime product\n(log scale)", fontsize=11)
    product_ax.set_xlabel("Number of vertices", fontsize=13, labelpad=8)
    if len(x) <= 25:
        product_ax.set_xticks(x)
    else:
        product_ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
    margin = max(0.5, 0.025 * np.ptp(x))
    product_ax.set_xlim(x[0] - margin, x[-1] + margin)
    for axis in (ax, product_ax):
        axis.grid(axis="y", color="#d9dee5", linewidth=1)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#6b7280")
        axis.tick_params(labelsize=10)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)

    def span(values):
        return str(min(values)) if min(values) == max(values) else f"{min(values)}–{max(values)}"

    parameters = ([rf"$c = {c:g}$"] if c is not None else [])
    parameters += [fr"{span(data['case_counts'])} filtrations per $n$",
                   f"{span(data['trials_per_case'])} trials per case"]
    figure_height = fig.get_figheight()
    fig.suptitle(r"Running times by number of vertices $n$ — " + ", ".join(parameters),
                 fontsize=14, fontweight="bold", y=1 - 0.10 / figure_height)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.40 / figure_height),
               ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.15, right=0.98, top=1 - 1.00 / figure_height,
                        bottom=0.58 / figure_height)
    fig.align_ylabels((ax, failure_ax, product_ax))
    # Match the left edge of each rotated title's first line, not its units.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_axes = (ax, failure_ax, product_ax)
    left_edges = [axis.yaxis.label.get_window_extent(renderer).x0 for axis in label_axes]
    for axis, left in zip(label_axes, left_edges):
        axis.yaxis.labelpad += (left - min(left_edges)) * 72 / fig.dpi
    return fig
