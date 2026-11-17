"""Field-dependence of Rips filtrations, and analysis of source of torsion.

We sample uniformly in [0,1]^n and retain 1/10 of all edges.
"""
# %% Imports

from collections import Counter
from datetime import datetime
import gc
import json
import math
from pathlib import Path
from statistics import fmean
from time import perf_counter

import gudhi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from _random_rips_plotting import save_figures, frequency_summary, source_summary
from _random_rips_run import run_trial, _uncomputed_module

# %% Functions

def run_experiment(n_points=1000, 
                   dimensions=(50, 75, 100), 
                   repetitions=5,
                   seed=0, 
                   edge_fraction=1/10,
                   output_dir=None):
    """Save tensor and conditionally computed free prime lists for every trial.

    Write config.json and trials.jsonl in output_dir, choosing a timestamped
    results directory if none is supplied. Flush each completed trial,
    including failures. Free is skipped when no tensor actual-radius changing
    primes were detected; a skipped list is None, not a computed empty list.
    Free failures preserve tensor results and subsequent trials continue.
    Interruptions propagate; earlier completed rows stay saved.
    """
    dimensions = list(dimensions)
    if (n_points < 2 or repetitions < 1 or not dimensions
            or any(d < 1 for d in dimensions) or len(set(dimensions)) != len(dimensions)
            or not 0 <= edge_fraction <= 1):
        raise ValueError("Use >= 2 points, positive repetitions/distinct dimensions, and an edge fraction in [0, 1]")
    target_edges = math.floor(edge_fraction * math.comb(n_points, 2))
    config = dict(schema_version=1,
                  n_points=n_points, dimensions=dimensions, repetitions=repetitions,
                  seed=seed, edge_fraction=edge_fraction, target_edges=target_edges,
                  edge_selection="shortest_edges_fixed_count",
                  edge_tie_break="lexicographic vertex pair",
                  simplex_order="GUDHI get_filtration(), including its tie order",
                  distribution="uniform [0,1]^d", homology_degree=1,
                  detector="native reduction / actual-radius and simplexwise H1 barcodes",
                  free_condition="tensor.rips_exceptional_primes is nonempty",
                  tensor_torsion="tensor.rips_betti_changing_primes: pointwise H1 dimensions differ from Q at actual radii",
                  map_torsion="free.rips_exceptional_primes: free H1 barcode differs from Q at actual radii",
                  source_denominator="trials with actual-radius field-coefficient H1 dependence",
                  sources_may_overlap=True,
                  nested_modules=["tensor", "free"], flat_detection_fields="tensor",
                  gudhi_version=gudhi.__version__, numpy_version=np.__version__)
    if output_dir is None:
        output_dir = Path("results") / ("torsion_rips_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite either file from an earlier run.
    for name in ("config.json", "trials.jsonl"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir / name}")
    with (output_dir / "config.json").open("x") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    log = (output_dir / "trials.jsonl").open("x")
    print(f"Saving trials to {output_dir / 'trials.jsonl'}", flush=True)
    results = []
    try:
        with tqdm(total=len(dimensions) * repetitions, desc="Rips H1 trials") as progress:
            for dimension in dimensions:
                for trial in range(repetitions):
                    progress.set_description(f"Rips H1, d={dimension}, trial={trial + 1}", refresh=False)
                    row = dict(dimension=dimension, trial=trial, seed=seed,
                               n_points=n_points, edge_fraction=edge_fraction,
                               target_edges=target_edges, edge_selection="shortest_edges_fixed_count")
                    started = perf_counter()
                    try:
                        row.update(run_trial(n_points, dimension, trial, edge_fraction, seed))
                    except Exception as error:
                        row.update(status="error", field_dependent=None,
                                   error=f"{type(error).__name__}: {error}")
                        row.setdefault("tensor", _uncomputed_module("error", "trial_failed"))
                        row.setdefault("free", _uncomputed_module("skipped", "tensor_not_completed"))
                    finally:
                        # Only small summaries survive a trial. Collect cycles
                        # before the next cloud, also after a failed reduction.
                        gc.collect()
                    row["seconds"] = perf_counter() - started
                    results.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    progress.set_postfix(status=row["status"], dependent=row["field_dependent"],
                                         free=row["free"]["status"], refresh=False)
                    progress.update()
    finally:
        log.close()
    return results

def _occurrence(record, key):
    """Require a completed result; unsuccessful trials never imply absence."""
    return bool(record[key]) if record.get("status") == "success" and record.get(key) is not None else None

def analyze_results(results):
    """Return dimension means, pooled frequencies and changing-prime counts."""
    frequencies, sources = frequency_summary(results), source_summary(results)
    output = {}
    for label, records, percent, count, denominator in (
        ("original_rips", frequencies, "rips_percent", "rips_count", "trials"),
        ("simplexwise_rips", frequencies, "simplexwise_percent", "simplexwise_count", "trials"),
        ("tensor_torsion", sources, "tensor_torsion_percent", "tensor_torsion_count", "affected_count"),
        ("map_torsion", sources, "map_percent", "map_count", "affected_count"),
    ):
        values = [row[percent] for row in records if row[percent] is not None]
        total = sum(row[denominator] for row in records)
        output[label] = dict(mean_percent=fmean(values) if values else None,
                             pooled_percent=100 * sum(r[count] for r in records) / total if total else None,
                             dimensions=len(values))
    affected = [row for row in results if row["tensor"]["rips_exceptional_primes"]]
    for label, rows, module, key in (
        ("original_rips", results, "tensor", "rips_exceptional_primes"),
        ("simplexwise_rips", results, "tensor", "simplexwise_exceptional_primes"),
        ("tensor_torsion", affected, "tensor", "rips_betti_changing_primes"),
        ("map_torsion", affected, "free", "rips_exceptional_primes"),
    ):
        prime_sets = [tuple(row[module][key]) for row in rows
                      if _occurrence(row[module], key) is not None]
        if len(prime_sets) != len(rows):
            raise ValueError(f"Incomplete experimental trial in {label}")
        positive = [primes for primes in prime_sets if primes]
        output[label].update(positive_trials=len(positive), trials=len(rows),
                             prime_count_distribution=dict(Counter(map(len, positive))),
                             prime_sets=[dict(primes=list(p), count=n) for p, n in sorted(Counter(positive).items())])
    output["cooccurrence"] = dict(both=sum(r["both_count"] for r in sources),
                                  tensor=sum(r["tensor_torsion_count"] for r in sources))
    return output

# %% Full parameters for the paper

SEED = 0
N_POINTS = 1000
DIMENSIONS = [3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
REPETITIONS = 1000
EDGE_FRACTION = 1/10

# %% Small parameters for testing 

SEED = 0
N_POINTS = 1000
DIMENSIONS = [40, 50]
REPETITIONS = 4
EDGE_FRACTION = 1/10

# %% Run and save

# Run the experiment
output = Path("results") / f"random_rips_{datetime.now():%Y%m%d_%H%M%S_%f}"
results = run_experiment(
    n_points=N_POINTS,
    dimensions=DIMENSIONS,
    repetitions=REPETITIONS,
    seed=SEED,
    edge_fraction=EDGE_FRACTION,
    output_dir=output,
)

# Analyze and save
if any(row["status"] != "success" for row in results):
    raise RuntimeError(f"Some trials failed; inspect {output / 'trials.jsonl'}")
analysis = analyze_results(results)
(output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
print(f"Saved Rips results to {output}")

# Plot
figures = save_figures(results, output)
plt.show()

# %% Open previous results and analyze

output = Path("results/random_rips_20261114_031510_091861")
with (output / "trials.jsonl").open() as file:
    results = [json.loads(line) for line in file if line.strip()]

# Print parameters
config = json.loads((output / "config.json").read_text())
print("Parameters:", {
    key: config[key]
    for key in ("n_points", "dimensions", "repetitions", "edge_fraction", "seed")
})

# Print analysis
analysis = analyze_results(results)
for name in ("original_rips", "simplexwise_rips", "tensor_torsion", "map_torsion"):
    data = analysis[name]
    print(f"\n{name.replace('_', ' ').capitalize()}:")
    print(f"  Positive trials: {data['positive_trials']} out of {data['trials']}")
    for label, key in (
        ("Mean across dimensions", "mean_percent"),
        ("Pooled frequency", "pooled_percent"),
    ):
        value = data[key]
        text = f"{value:.2f}%" if value is not None else "N/A"
        print(f"  {label}: {text}")
    print("  Number of primes → number of positive trials:",
          data["prime_count_distribution"])
    for entry in data["prime_sets"]:
        print(f"  Prime set {entry['primes']}: {entry['count']} trials")

print("\nTrials with both tensor and map torsion:",
      analysis["cooccurrence"]["both"])

"""
Parameters: {'n_points': 1000, 'dimensions': [3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'repetitions': 1000, 'edge_fraction': 0.1, 'seed': 0}

Original rips:
  Positive trials: 455 out of 13000
  Mean across dimensions: 3.50%
  Pooled frequency: 3.50%
  Number of primes → number of positive trials: {1: 446, 2: 9}
  Prime set [2]: 444 trials
  Prime set [2, 3]: 9 trials
  Prime set [3]: 2 trials

Simplexwise rips:
  Positive trials: 487 out of 13000
  Mean across dimensions: 3.75%
  Pooled frequency: 3.75%
  Number of primes → number of positive trials: {1: 478, 2: 9}
  Prime set [2]: 476 trials
  Prime set [2, 3]: 9 trials
  Prime set [3]: 2 trials

Tensor torsion:
  Positive trials: 60 out of 455
  Mean across dimensions: 13.63%
  Pooled frequency: 13.19%
  Number of primes → number of positive trials: {1: 60}
  Prime set [2]: 60 trials

Map torsion:
  Positive trials: 405 out of 455
  Mean across dimensions: 88.13%
  Pooled frequency: 89.01%
  Number of primes → number of positive trials: {1: 398, 2: 7}
  Prime set [2]: 394 trials
  Prime set [2, 3]: 7 trials
  Prime set [3]: 4 trials

Trials with both tensor and map torsion: 10
"""
