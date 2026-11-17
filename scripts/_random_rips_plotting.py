import json
import math
from pathlib import Path
import numpy as np

def _module_record(row, name):
    return row.get(name) or {}

def _occurrence(record, key):
    """Require a completed result; unsuccessful trials never imply absence."""
    return bool(record[key]) if record.get("status") == "success" and record.get(key) is not None else None

def _require_occurrence(row, module, key, label):
    decision = _occurrence(_module_record(row, module), key)
    if decision is None:
        raise ValueError(
            f"Cannot determine {label} for dimension {row['dimension']}, "
            f"trial {row.get('trial', '?')}; compute this result before plotting. "
        )
    return decision

def _trial_groups(results):
    """Validate comparable campaigns and retain every supplied trial."""
    if not results:
        raise ValueError("No trials to plot")
    if any(row.get("simplexwise_refinement", False) for row in results):
        raise ValueError("These records do not contain a separate actual-Rips comparison")
    for key, default in (("edge_selection", "calibrated_radius"),
                         ("n_points", None), ("edge_fraction", None)):
        if len({row.get(key, default) for row in results}) != 1:
            raise ValueError(f"Plot trials with different {key} values separately")
    if any("dimension" not in row or "n_points" not in row for row in results):
        raise ValueError("Every trial must record its dimension and n_points")
    return [(dimension, [row for row in results if row["dimension"] == dimension])
            for dimension in sorted({row["dimension"] for row in results})]

def frequency_summary(results):
    """Count original and simplexwise field dependence over all trials.

    Every completed native result is exact and exhaustive. Failed or unrun
    results raise rather than changing the denominator. Free-stage failures
    do not erase completed field-coefficient results.
    """
    summaries = []
    for dimension, group in _trial_groups(results):
        rips = refined = 0
        for row in group:
            original = _require_occurrence(row, "tensor", "rips_exceptional_primes",
                                           "original Rips field dependence")
            simplexwise = _require_occurrence(row, "tensor", "simplexwise_exceptional_primes",
                                             "simplexwise field dependence")
            if original and not simplexwise:
                raise ValueError("Original Rips dependence must survive simplexwise refinement")
            rips += original
            refined += simplexwise
        summaries.append(dict(dimension=dimension, trials=len(group),
                              rips_count=rips, simplexwise_count=refined,
                              rips_percent=100 * rips / len(group),
                              simplexwise_percent=100 * refined / len(group)))
    return summaries

def source_summary(results):
    """Classify sources conditional on actual-radius field-coefficient changes.

    Tensor torsion means a pointwise H1 dimension differs from Q; map torsion
    means the free H1 barcode differs from Q. Both tests quantify existence
    over primes and original radii. Their percentages share the affected-trial
    denominator and can overlap, even through different primes in one trial.
    Return None percentages when this denominator is zero. Unknown membership
    or classification raises, so failures never silently change the sample.
    """
    summaries = []
    for dimension, group in _trial_groups(results):
        affected = maps = tensor_torsions = both = 0
        for row in group:
            if not _require_occurrence(row, "tensor", "rips_exceptional_primes",
                                       "original Rips field dependence"):
                continue
            affected += 1
            tensor_torsion = _require_occurrence(row, "tensor", "rips_betti_changing_primes",
                                        "tensor torsion at original Rips radii")
            map_torsion = _require_occurrence(row, "free", "rips_exceptional_primes",
                                              "map torsion at original Rips radii")
            if not (tensor_torsion or map_torsion):
                raise ValueError("An affected H1 filtration must have tensor or map torsion")
            maps += map_torsion
            tensor_torsions += tensor_torsion
            both += map_torsion and tensor_torsion
        summaries.append(dict(
            dimension=dimension, trials=len(group), affected_count=affected,
            map_count=maps, tensor_torsion_count=tensor_torsions, both_count=both,
            map_percent=100 * maps / affected if affected else None,
            tensor_torsion_percent=100 * tensor_torsions / affected if affected else None,
            both_percent=100 * both / affected if affected else None,
        ))
    return summaries

def _style_percentage_axes(ax, dimensions, positions, *, upper=105):
    """Share the two figures' dimension ordering and percentage formatting."""
    from matplotlib.ticker import PercentFormatter

    ax.set_xticks(positions, labels=[str(d) for d in dimensions])
    # NaN bars do not contribute to autoscaling; reserve all dimension groups.
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.set_xlabel("Ambient dimension", fontsize=14)
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

def _label_bars(ax, bars):
    """Round displayed percentages only; saved summaries retain exact counts."""
    ax.bar_label(bars, labels=[f"{bar.get_height():.1f}".rstrip("0").rstrip(".") + "%"
                              if math.isfinite(bar.get_height()) else "" for bar in bars],
                 padding=4, fontsize=9)

def _save_figure(fig, output_path):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)

def plot_results(results, output_path=None, *, figsize=(12, 8), annotate=True):
    """Plot original/refined field-coefficient H1 frequencies as vertical bars.

    Every bar uses all trials at its dimension. Original Rips is blue;
    simplexwise Rips is orange. Return (figure, axis). Source classification
    is independent of this plot.
    """
    import matplotlib.pyplot as plt

    summaries = frequency_summary(results)
    positions = np.arange(len(summaries)) * 1.25
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.77)
    width, gap = 0.30, 0.20
    for offset, key, label, color in (
        (-(width + gap) / 2, "rips_percent", "Original Rips filtration", "#1f77b4"),
        ((width + gap) / 2, "simplexwise_percent", "Simplex-wise Rips filtration", "#ff7f0e"),
    ):
        bars = ax.bar(positions + offset, [row[key] for row in summaries], width,
                      label=label, color=color)
        if annotate:
            _label_bars(ax, bars)
    maximum = max(max(row["rips_percent"], row["simplexwise_percent"]) for row in summaries)
    upper = min(110, max(5, 5 * (math.floor(maximum / 5) + 1)))
    _style_percentage_axes(ax, [row["dimension"] for row in summaries], positions, upper=upper)
    ax.set_ylabel(r"Filtrations with field-dependent $H_1$ (%)", fontsize=14)
    counts = [row["trials"] for row in summaries]
    trials_label = (f"{counts[0]} filtrations per dimension" if len(set(counts)) == 1
                    else f"{min(counts)}–{max(counts)} filtrations per dimension")
    edge_count = results[0].get("edges", results[0].get("target_edges"))
    if edge_count is None:
        edge_count = math.floor(results[0]["edge_fraction"] * math.comb(results[0]["n_points"], 2))
    fig.suptitle(r"Field-dependent $H_1$-barcodes in Rips filtrations",
                 fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.905,
             f"{results[0]['n_points']} points per cloud, {trials_label}, {edge_count:,} edges",
             ha="center", va="top", fontsize=13)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.85), ncol=2, frameon=False, fontsize=12)
    _save_figure(fig, output_path)
    return fig, ax

def plot_torsion_sources(results, output_path=None, *, figsize=(12, 8), annotate=True):
    """Plot tensor/map occurrence among affected original-Rips filtrations.

    Counts k appear directly below each dimension tick and give the conditional
    denominator. Grouped bars allow overlapping sources;
    a zero denominator is marked unavailable, never plotted as a measured zero
    percent. Return (figure, axis).
    """
    import matplotlib.pyplot as plt

    summaries = source_summary(results)
    positions = np.arange(len(summaries)) * 1.25
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.21, top=0.84)
    width, gap = 0.30, 0.20
    for offset, key, label, color in (
        (-(width + gap) / 2, "tensor_torsion_percent", r"Tensor torsion, in pointwise groups $H_1(K^t;\mathbb{Z})$", "#2ca02c"),
        ((width + gap) / 2, "map_percent", r"Map torsion, in maps $H_1^\mathrm{free}(K^s;\mathbb{Z})\to H_1^\mathrm{free}(K^t;\mathbb{Z})$", "#222222"),
    ):
        values = [row[key] if row[key] is not None else np.nan for row in summaries]
        bars = ax.bar(positions + offset, values, width, label=label, color=color)
        if annotate:
            _label_bars(ax, bars)
    _style_percentage_axes(ax, [row["dimension"] for row in summaries], positions, upper=112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Ambient dimension\n" + r"($k$ = number of filtrations)",
                  fontsize=14, linespacing=1.5, labelpad=12)
    ax.set_xticks(positions, labels=[str(row["dimension"]) + "\n"
                                    + rf"$(k = {row['affected_count']})$"
                                    for row in summaries])
    for position, row in zip(positions, summaries):
        if row["affected_count"] == 0:
            ax.text(position, 8, "No affected\nfiltrations", ha="center", va="bottom",
                    fontsize=9, color="0.45", style="italic")
    ax.set_ylabel(r"Filtrations with pointwise or map torsion in $H_1$ (%)", fontsize=14)
    fig.suptitle(r"Sources of torsion in $H_1$-barcodes of Rips filtrations",
                 fontsize=15, fontweight="bold", y=0.97)
    # fig.text(0.5, 0.89, "Among filtrations whose original-Rips field-coefficient H₁ barcode depends on the field",
    #          ha="center", fontsize=10, color="0.35")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False, fontsize=12)
    _save_figure(fig, output_path)
    return fig, ax

def save_figures(results, output_dir):
    """Write both figures as PNG/PDF plus their counts and denominators as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(field_dependence=frequency_summary(results),
                   torsion_sources=source_summary(results), sources_may_overlap=True)
    figures = (plot_results(results), plot_torsion_sources(results))
    for name, (fig, _) in zip(("field_dependence", "torsion_sources"), figures):
        for extension in ("png", "pdf"):
            _save_figure(fig, output_dir / f"{name}.{extension}")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return figures
