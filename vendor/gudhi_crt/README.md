# Patched GUDHI CRT competitor

This optional backend reproduces the CRT method used by
`scripts/comparison_competitors.py`. It is separate from the installed GUDHI
Python package used to construct filtrations and run the brute-force method.

From the package root, build explicitly:

```sh
python vendor/gudhi_crt/build.py --test
```

The build requires a C++17 compiler, Boost headers, GMP and FLINT development
libraries, and `patch`. It verifies `source_manifest.json`, copies the pinned
headers into the ignored `build/` directory, applies both patches and records
compiler/source/patch/executable hashes in `build/build.json`. `--test` also
checks CRT projectors, canonical zero, both sparse insertion paths and reset
behavior. Importing the Python wrapper does not compile or launch anything.

`vendor/` is the unmodified MIT-licensed GUDHI 3.6.0 header subset from commit
`de8bd5109fcdc6d4d200c74685bab031d953d2af`; see its `LICENSE` and the source
manifest. The driver, build script, tests and numbered patches are local
integration material. The first patch accepts exactly the requested GMP
integer primes and resets CRT initialization. The second fixes canonical zero
and avoids storing zero coefficients in both sparse insertion branches.
The driver proves primes with FLINT before calling the patched initializer;
the initializer's additional GMP screening is not used as a proof.

The Python entry point is `scripts/_gudhi_crt.py:run_native`. It transports
integer simplex indices and prime identifiers without narrowing primes. The
native driver verifies count/order and face-before-coface validity. Repeated
intervals retain multiplicity. The output timer covers requested-degree pair
extraction and exact field-mask expansion; cleanup, sorting and serialization
are separate. Computation includes maximal-dimensional homology by default;
the comparison script omits H2 on triangle inputs while retaining H1 on graphs.

The wrapper enforces a wall timeout, disables core dumps and optionally caps
the native child's address space (default 1024 MiB). It reports failed trials
as failures rather than empty successful barcodes. Its child remains in the
comparison worker's process group so the worker watchdog can stop both.
