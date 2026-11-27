"""Benchmark of four methods on 2-dimensional Linial-Meshulam filtrations.

- brute_force: GUDHI via Python interface
- modular:     modular reconstruction via CRT in C++ GUDHI implementation (under vendor/gudhi_crt)
- omnifield:   omni-field persistence via Dionysus
- ours:        integral_persistence (under /cpp)

CRT is patched to accept exactly the discovered primes (instead of all primes in an interval). 

Benchmark 1: compares all four methods at a fixed vertex count.
Benchmark 2: compares modular reconstruction and ours across vertex counts.
"""

# %% Imports
# %cd scripts

from pathlib import Path
import math
import json
from datetime import datetime

from tqdm import tqdm

import integral_persistence as ip
from _benchmark_filtrations import generate_triangle_order, build_filtration
from _benchmark_plotting import plot_comparison_results, plot_comparison_curves, plot_comparison_by_vertices, vertex_comparison_data
from _benchmark_methods import run_brute_force, run_modular, run_omnifield, run_ours, run_fresh_trial, summarize_trials

# %% Functions

TRIAL_TIMEOUT_SECONDS = 10.0
METHOD_TIMEOUT_SECONDS = {}
MODULAR_MEMORY_MB = 1024  # Memory cap for modular reconstruction
_DEFAULT_MODULAR_MEMORY = object()

def run_trial_fresh(method, order, model="LM", timeout=None, on_timeout="raise",
                    *, memory_mb=_DEFAULT_MODULAR_MEMORY, on_error="record"):
    """Run one complete method with a configurable whole-worker time limit."""
    if timeout is None:
        timeout = METHOD_TIMEOUT_SECONDS.get(method, TRIAL_TIMEOUT_SECONDS)
    if memory_mb is _DEFAULT_MODULAR_MEMORY:
        memory_mb = MODULAR_MEMORY_MB
    return run_fresh_trial(
        method, order, model=model, timeout=timeout,
        on_timeout=on_timeout, modular_memory_mb=memory_mb, on_error=on_error,
    )

def run_benchmark(n, count, repetitions, trials, RUNNERS=None, max_prime=46337):
    """Run fresh trials with optional preliminary prime-bound screening.

    With max_prime=None, omit parent discovery entirely. Completed workers
    using canonical discovery provide reference candidates when available;
    otherwise reference_primes remains None, not an empty measured list.
    A numeric cutoff retains screening before launching any workers.
    """

    if RUNNERS is None:
        RUNNERS = {
            "brute_force": run_brute_force,
            "modular": run_modular,
            "omnifield": run_omnifield,
            "ours": run_ours,
        }
    RESULTS = {method: [] for method in RUNNERS}
    RAW_RESULTS = {method: [] for method in RUNNERS}
    INPUTS = []
    repetitions_successful = 0
    next_seed = 0

    with tqdm(total=repetitions * trials * len(RUNNERS), unit="run") as progress:
        while repetitions_successful < repetitions:
            seed = next_seed
            next_seed += 1  # Advance even if this filtration is rejected

            progress.set_description(f"Seed {seed}: generating")
            order = generate_triangle_order(n, count, seed=seed)

            primes = None
            if max_prime is not None:
                progress.set_description(f"Seed {seed}: screening primes")
                simplices, times = build_filtration(order, model="LM")
                data = ip.build_matrices(simplices, times=times, degrees=[1], modules=["field"])[1]["field"]
                discovery = ip.rational_reduction(data.matrix)
                primes = discovery.primes
                # Screening is outside timing and cannot warm a fresh worker.
                del discovery, data, simplices, times
                if max(primes, default=0) > max_prime:
                    progress.write(
                        f"Seed {seed}: skipping unsupported primes: {primes}"
                    )
                    continue

            INPUTS.append(order.to_dict())
            measurements = {method: [] for method in RUNNERS}
            methods = tuple(RUNNERS)
            for trial in range(trials):
                # Rotate positions to spread slow changes in system load or clocks.
                offset = (repetitions_successful + trial) % len(methods)
                for method in methods[offset:] + methods[:offset]:
                    progress.set_description(
                        f"Seed {seed}: {method}, trial {trial + 1}/{trials}"
                    )
                    record = run_trial_fresh(method, order, on_timeout="record")
                    record["seed"] = seed
                    record["trial"] = trial
                    measurements[method].append(record)
                    if record["status"] == "timeout":
                        progress.write(
                            f"Seed {seed}: {method}, trial {trial + 1}/{trials}: "
                            f"timed out after {record['timeout_seconds']:g} s; continuing."
                        )
                    elif record["status"] in ("memory_limit", "error"):
                        label = "memory allocation failed" if record["status"] == "memory_limit" else "worker failed"
                        progress.write(f"Seed {seed}: {method}, trial {trial + 1}/{trials}: {label}; continuing.")
                    progress.update(1)

            # Reuse completed discovery evidence; never launch another pass.
            # Omnifield may report additional candidates, so cannot supply this
            # canonical reference. A failed trial's prime list stays unknown.
            for method in ("brute_force", "modular", "ours"):
                for record in measurements.get(method, ()):
                    if record["status"] != "success":
                        continue
                    candidates = tuple(sorted(record["primes"]))
                    if primes is None:
                        primes = candidates
                    elif candidates != tuple(sorted(primes)):
                        raise ValueError(f"Seed {seed}: canonical prime discovery disagrees between runs")

            for method in methods:
                samples = measurements[method]
                RAW_RESULTS[method].append(samples)
                averaged = summarize_trials(
                    method, samples, seed=seed, order=order, reference_primes=primes,
                )
                RESULTS[method].append(averaged)

            repetitions_successful += 1

    print(
        f"Completed {repetitions_successful} accepted filtrations "
        f"from {next_seed} candidates, with {trials} trials per method."
    )

    return RESULTS, RAW_RESULTS, INPUTS

# %% ------ EXAMPLE

# Parameters
n = 30
c = 2.7538
count = int((2*c/n)*math.comb(n, 3))
seed = 1
order = generate_triangle_order(n, count, seed=seed)

# Run
RUNNERS = {"brute_force": run_brute_force, "modular": run_modular, "omnifield": run_omnifield, "ours": run_ours}
results = {method: run_trial_fresh(method, order, on_timeout="record") for method in RUNNERS}

# Print phases
print("Phase timings (seconds):")
for method, result in results.items():
    print(f"    {method:<12}: ", result["phases_seconds"])

"""
Phase timings (seconds):
    brute_force :  {'prime_discovery': 0.0036868219999999998, 'method_initialization': 0.0005047409999860974, 'main_computation': 0.001613689000123486, 'output': 0.0}
    modular     :  {'prime_discovery': 0.002473561, 'method_initialization': 0.000731794, 'main_computation': 0.003373096, 'output': 3.5205e-05}
    omnifield   :  {'prime_discovery': 2.0200013750582002e-07, 'method_initialization': 0.0009720549999201467, 'main_computation': 0.011107867000191618, 'output': 0.0003345879999869794}
    ours        :  {'prime_discovery': 0.0023536, 'method_initialization': 0.000641806, 'main_computation': 0.000103566, 'output': 5.098e-06}
"""

# Print primes
print("\nDiscovered primes:")
for method, result in results.items():
    print(f"    {method:<12}: ", result["primes"])

"""
Discovered primes:
    brute_force :  (2,)
    modular     :  (2,)
    omnifield   :  (23, 17, 13, 11, 19, 7, 5, 3, 2)
    ours        :  (2,)
"""

# %% ------ BENCHMARK 1: All methods

# %% Small parameters for testing 

c = 2.7538
REPETITIONS = 3
TRIALS = 2
NUMBER_VERTICES = [30,50]
TRIAL_TIMEOUT_SECONDS = 10.0

# %% Full parameters for the paper

c = 2.7538
REPETITIONS = 100
TRIALS = 5
NUMBER_VERTICES = [30, 40, 50]
TRIAL_TIMEOUT_SECONDS = 60.0

# %% Run

output_directory = Path("results") / f"benchmark1_{datetime.now():%Y%m%d_%H%M%S_%f}"
output_directory.mkdir(parents=True, exist_ok=True)

for n in NUMBER_VERTICES:
    # Run
    count = int((2*c/n)*math.comb(n, 3))
    RESULTS, RAW_RESULTS, INPUTS = run_benchmark(n, count, REPETITIONS, TRIALS)

    # Save results
    parameters = f"n{n}-repetitions{REPETITIONS}-trials{TRIALS}"
    results_path = output_directory / f"results-{parameters}.json"
    results_path.write_text(json.dumps({"inputs": INPUTS, "results": RESULTS, "raw_results": RAW_RESULTS}, indent=2) + "\n")

    # Plot
    fig_average, fig_by_primes = plot_comparison_results(RESULTS, n=n, count=count, trials=TRIALS)
    CURVE_HEIGHT_INCHES = 6.65 if n>=50 else 5.6
    fig_curves = plot_comparison_curves(RESULTS, n=n, count=count, trials=TRIALS, height=CURVE_HEIGHT_INCHES)

    # Save figures
    fig_average.savefig(output_directory / f"running_time_average-{parameters}.pdf", dpi=250, bbox_inches="tight")
    fig_by_primes.savefig(output_directory / f"running_time_by_primes-{parameters}.pdf",dpi=250,bbox_inches="tight")
    fig_curves.savefig(output_directory / f"running_time_curves-{parameters}.pdf", dpi=250, bbox_inches="tight")

# %% --- BENCHMARK 2: Against modular reconstruction only

# %% Small parameters for testing 

c = 2.7538
RUNNERS = {"modular": run_modular, "ours": run_ours}
NUMBER_VERTICES = [20,30,40]
REPETITIONS = 5
TRIALS = 2
TRIAL_TIMEOUT_SECONDS = 60

# %% Full parameters for the paper

c = 2.7538
RUNNERS = {"modular": run_modular, "ours": run_ours}
NUMBER_VERTICES = range(10,90+1,10)
REPETITIONS = 100
TRIALS = 2
TRIAL_TIMEOUT_SECONDS = 60

# %% Run 

# Run
RESULTS_ALL = []
RAW_RESULTS_ALL = []
INPUTS_ALL = []
with tqdm(NUMBER_VERTICES, desc="Overall", unit="size", position=0, smoothing=0, dynamic_ncols=True) as progress:
    for n in progress:
        count = int((2*c/n) * math.comb(n, 3))
        progress.set_postfix(n=n, triangles=count, simplices=n + math.comb(n, 2) + count)
        RESULTS, RAW_RESULTS, INPUTS = run_benchmark(n, count, REPETITIONS, TRIALS, RUNNERS=RUNNERS, max_prime=None)
        RESULTS_ALL.append(RESULTS)
        RAW_RESULTS_ALL.append(RAW_RESULTS)
        INPUTS_ALL.append(INPUTS)

# Save results
output_directory = Path("results") / f"benchmark2_{datetime.now():%Y%m%d_%H%M%S_%f}"
output_directory.mkdir(parents=True, exist_ok=True)
parameters = f"repetitions{REPETITIONS}-trials{TRIALS}-timeout{TRIAL_TIMEOUT_SECONDS:g}s"
results_path = output_directory / f"results-{parameters}.json"
results_path.write_text(json.dumps({"inputs": INPUTS, "results": RESULTS, "raw_results": RAW_RESULTS}, indent=2) + "\n")
results_path.write_text(json.dumps({
    "vertices": list(NUMBER_VERTICES)[:len(RESULTS_ALL)],
    "c": c,
    "repetitions": REPETITIONS,
    "trials": TRIALS,
    "trial_timeout_seconds": TRIAL_TIMEOUT_SECONDS,
    "methods": list(RUNNERS),
    "max_prime": None,
    "modular_memory_mb": MODULAR_MEMORY_MB,
    "phase_average": "completed_trials_only",
    "results": RESULTS_ALL,
    "raw_results": RAW_RESULTS_ALL,
    "inputs": INPUTS_ALL,
}, indent=2), encoding="utf-8")
print(f"Saved to {results_path}")

# Plot and save figure
fig_vertices = plot_comparison_by_vertices(
    RESULTS_ALL,
    vertices=list(NUMBER_VERTICES)[:len(RESULTS_ALL)],
    trials=TRIALS, c=c, time_unit="s", height=7.0, memory_limit_mb=MODULAR_MEMORY_MB,
)
fig_vertices.savefig(output_directory / f"running_time_by_vertices-{parameters}.pdf", dpi=250, bbox_inches="tight", pad_inches=0.05)

# Analyse results

summary = vertex_comparison_data(
    RESULTS_ALL,
    vertices=list(NUMBER_VERTICES)[:len(RESULTS_ALL)],
    trials=TRIALS,
)
j = summary["vertices"].index(90)
m = summary["methods"].index("ours")
mean_seconds = summary["mean_seconds"][m][j]
print(f"Our method at n=90: {mean_seconds:.3f} s") # Our method at n=90: 25.168 s

# %% ------ EXPERIMENT: Large torsion at n=70

n = 70
repetitions = 100
c = 2.7538
count = int((2*c/n)*math.comb(n, 3))

PRIMES = []
with tqdm(total=repetitions, unit="run") as progress:
    for seed in range(repetitions):
        progress.set_description(f"Seed {seed}: prime discovery")

        order = generate_triangle_order(n, count, seed=seed)
        simplices, times = build_filtration(order, model="LM")
        data = ip.build_matrices(simplices, times=times, degrees=[1], modules=["field"])[1]["field"]
        discovery = ip.rational_reduction(data.matrix)
        PRIMES.append(discovery.primes)
        del discovery, data, simplices, times
        progress.update(1)

# Largest primes found
nb_digits = [math.log10(math.prod(p)) for p in PRIMES]
largest_prime = [max(p, default=0) for p in PRIMES]
print("Largest prime found:", max(largest_prime, default=None))
print("Largest number of digits in a product:", max(nb_digits))

# Largest prime found: 88823020457549174488262651
# Largest number of digits in a product: 2510.939186544896

# %% ------- EXPERIMENT: Large torsion at n=70

seed = 3

n = 75
c = 2.7538
count = int((2*c/n)*math.comb(n, 3))
order = generate_triangle_order(n, count, seed=seed)
print("Number triangles =", count)

simplices, times = build_filtration(order, model="LM")
data = ip.build_matrices(simplices, times=times, degrees=[1], modules=["field"])[1]["field"]
state = ip.reduction(data.matrix)
original_bars = ip.decode_barcodes(state, data)
simplexwise_bars = ip.decode_barcodes(state, data, times=list(range(len(simplices))))

print("Exceptional primes:", original_bars.exceptional_primes)
print("Largest dependent prime:", max(original_bars.exceptional_primes, default=None))
print("Largest simplexwise prime:", max(simplexwise_bars.exceptional_primes, default=None))

# Largest dependent prime:                  33483214385461 (seed 1)
# Largest dependent prime:             2169334844701017187 (seed 3)
# Largest simplexwise prime:   528396606216453961845426307 (seed 1)
# Largest simplexwise prime: 10681992134913386783887592587 (seed 3)
# Compare with:                 79040679454167077902597570 (not prime, Kahle–Lutz–Newman–Parsons)
# Compare with:                 80555235907994145009690263 (prime, https://gap-packages.github.io/hap/tutorial/chap5.html)
