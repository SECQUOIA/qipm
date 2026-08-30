# Quantum Interior Point Methods (QIPM)

Companion code for the IEEE Quantum Week 2026 submission--[arXiv](https://arxiv.org/abs/2604.24362).
Please cite this paper as:
```
@misc{Binkowski2026PracticalLowerBoundsForHybridQuantumInteriorPointMethodsInLinearProgramming,
    author          = {Binkowski, Lennart},
    year            = {2026},
    title           = {Practical lower bounds for hybrid quantum interior point methods in linear programming},
    eprint          = {2604.24362},
    archivePrefix   = {arXiv},
    primaryClass    = {quant-ph},
    url             = {https://arxiv.org/abs/2604.24362}
}
```

Reports a quantum-favorable screening estimate hierarchy for a specified hybrid
QIPM pipeline applied to linear programs (LPs) of the form

$$\min\ c^\top x \quad \text{s.t.}\ Ax = b,\ x \geq 0$$

via quantum interior-point methods (QIPMs).

## Pipeline

```
extract → transform → solve → benchmark
```

Each stage writes results into `cache_dir/<class>/<name>/`:

| File | Contents |
|------|----------|
| <name>.mps | Original MPS file |
| <name>.std | Standard-form LP (NPZ: sparse `A`, vectors `b`, `c`, scalar `obj_offset`) |
| <name>.data | JSON accumulating cycle counts, sparsity, condition numbers, runtimes |

Results accumulate across stages. Re-running transform when the decompressed `.std` arrays are unchanged preserves downstream results; generating different arrays purges those fields before replacing `.std`, and removing `.std` clears them as well.

Instance classes: `clique`, `independent_set`, `max_flow`, `miplib`, `misc`, `netlib`, `stochlp`, `vertex_cover`.

## Usage

### 1. Extract

Populate `cache_dir` from the simplex-benchmarks repository (requires Git LFS):

```bash
python extract.py                              # clone automatically, then clean up
python extract.py /path/to/simplex-benchmarks  # use existing clone
```

Without an argument, the script performs a shallow clone (`--depth 1`) of the repository, pulls LFS objects, extracts the relevant data, then deletes the temporary clone. Passing a path skips the clone entirely.

Two types of data are extracted:

**MPS instances** — each zip in `mps/` is unpacked and its `.mps` files (from `min/` or `max/` subdirectories) are placed under `cache_dir/<class>/<stem>/`. Instance classes are assigned as follows:

| Zip | Class |
|-----|-------|
| netlib, miplib, stochlp, misc | same name |
| max_flow, random_directed_graphs | max_flow |
| clq_mis_vc_dimacs, clq_mis_vc_random | by filename prefix: `clq`→clique, `is`→independent_set, `vc`→vertex_cover |

**GLPK runtimes** — the evaluation results in `benchmark/01_evaluation/` contain compressed archives of `.data` files with per-instance GLPK solve times (`runtime_primal`). These are merged into each instance's `.data` file as `runtime_glpk`, preserving results from other stages. Fresh clones also produce `cache_dir/extract_manifest.json` with the source commit and extraction time.

### 2. Transform

Convert MPS instances to standard-form LP via HiGHS presolve:

```bash
python transform.py                        # all instance classes
python transform.py netlib miplib          # selected classes
python transform.py --cache-dir /my/cache
```

Each MPS file is first presolved by HiGHS, then the reduced LP is algebraically converted to standard form $\min c^\top x$ s.t. $Ax = b,\ x \geq 0$. The conversion handles all bound and row types:

| Variable type | Transformation |
|---|---|
| Bounded $l \leq x \leq u$ | $x = l + x_1$, add row $x_1 + s = u - l$,\ $x_1, s \geq 0$ |
| Lower-bounded $l \leq x$ | Shift $x_1 = x - l \geq 0$ |
| Upper-bounded $x \leq u$ | Negate $x_1 = u - x \geq 0$ |
| Free $x \in \mathbb{R}$ | Split $x = x^+ - x^-$,\ $x^+, x^- \geq 0$ |

| Row type | Treatment |
|---|---|
| Equality $Ax = b$ | Kept as-is |
| $\leq$ inequality | Add slack $s \geq 0$ |
| $\geq$ inequality | Add surplus $s \geq 0$ (negated) |
| Range $l \leq Ax \leq u$ | Add slack for upper bound; extra row $s_1 + s_2 = u - l$ |
| Free row | Dropped entirely |

After conversion, harmless zero rows with zero right-hand sides are removed. A zero row with a nonzero right-hand side is recorded as infeasible. The result is saved as a compressed NPZ file with extension `.std`. Variable shifts and the original MPS objective offset are stored as `obj_offset`, so $z_{\mathrm{mps}} = z_{\mathrm{std}} + \mathtt{obj\_offset}$.

The transform stage writes `transform_status` (`ok`, `reduced_to_empty`, `infeasible`, `unbounded`, `unbounded_or_infeasible`, or `error:<ExceptionName>`) to `.data`. An unchanged successful transform preserves downstream results. A changed successful transform clears them; every non-success conversion outcome removes a stale `.std` and clears them. Stage `--show` commands report successful totals with a breakdown of non-success statuses.

### 3. Solve (optional)

Solve instances with HiGHS and record solve time:

```bash
python solve.py                           # all classes, both formats
python solve.py --format std netlib       # .std only, netlib class
python solve.py --format mps              # .mps only
```

Each instance is solved in two independent modes, controlled by `--format`:

| Format | Input | HiGHS model | Output key |
|--------|-------|-------------|------------|
| mps | Raw `.mps` | Read directly, HiGHS selects solver | runtime_highs_mps |
| std | NPZ `.std` | Equality LP $(Ax = b,\ x \geq 0)$ reconstructed from $A, b, c$ | runtime_highs_std |

For `.std`, if the default solver fails (e.g. due to poor scaling), the solve is automatically retried with HiGHS's interior-point method.

Each solve runs in a subprocess with a 10-minute timeout. HiGHS is configured with `threads=1` before every solve; when `solve.py` starts the process, it also defaults the OpenMP, OpenBLAS, and MKL thread-count environment variables to 1 unless the user has set them explicitly. In `both` mode, if the `.mps` solve times out, the `.std` solve is recorded as `skipped_mps_timeout` without a runtime. The recorded wall time covers `Highs.run()` only and excludes model construction. It is written to the instance's `.data` JSON and serves as the classical baseline for the screening comparison.

`solve_status_mps` and `solve_status_std` record `ok`, `ok_ipm`, `timeout`, `crashed`, `non_optimal`, `skipped_mps_timeout`, or `error:<ExceptionName>`. Runtime keys are present only for optimal solves. The top-level `highs_version` and `highs_threads` keys record solve provenance; the last solve-stage run for either format wins.

### 4. Benchmark

Compute benchmark-model-2 screening estimates and write them to `.data`:

```bash
python benchmark.py                           # all classes, both variants
python benchmark.py --variant mnes            # MNES only
python benchmark.py --variant oss netlib      # OSS, netlib class only
python benchmark.py --cache-dir /my/cache
```

`--variant` accepts `mnes`, `oss`, or `both` (default).

Two QIPM variants are benchmarked:

| Variant | System | Matrix | Size |
|---------|--------|--------|------|
| mnes | Modified Normal Equation System | $\hat{M} = I + \bar{F}\bar{F}^\top$ | $m \times m$ |
| oss | Orthogonal Subspaces System | $M = [-A^\top \mid V]$ | $n \times n$ |

For each instance, the script reads $A$ and $b$ from the `.std` file; $b$ feeds the dropped-row consistency check. The top-level `benchmark_model` record key is `2`. The script writes these additional keys for each variant (`v` is `mnes` or `oss`):

| Key | Meaning |
|---|---|
| `cycle_count_v` | Modeled Chebyshev pipeline total |
| `cycle_count_best_known_v` | Best-known-solver estimate total |
| `cycle_count_floor_v` | Scaling-only floor total; unknown $\Omega$ constant set to 1 |
| `sparsity_v` | Sampled or enumerated underestimate of QLSA sparsity $s$ |
| `sparsity_method_v` | How $s$ was obtained: `exact` or `sampled` |
| `cond_v` | Condition-number bound $\kappa$ |
| `cond_method_v` | How $\kappa$ was obtained: `exact`, `svds`, `probe_sigma_max`, `probe_sigma_min`, or `probe_both` |
| `qlsa_queries_v` | Modeled Chebyshev per-solve query count |
| `qlsa_queries_best_known_v` | Best-known per-solve query estimate |
| `qlsa_queries_floor_v` | Scaling-only per-solve floor |
| `tomography_reps_v` | Tomography repetition count $R$ |

Each total obeys the exact invariant `cycle_count_x_v = qlsa_queries_x_v × tomography_reps_v`, with the empty `x` suffix denoting the modeled line.

`status_mnes` and `status_oss` record `ok`, `timeout`, `crashed`, `skipped_too_large`, `skipped_degenerate`, `rank_uncertain`, `inconsistent_rows`, or `error:<ExceptionName>`. Instances with more than 100,000 rows are recorded as `skipped_too_large`, so the largest instances are excluded from the screened population. `benchmark.py --show` labels old successful records and statusless records carrying a finite legacy count `outdated_model`; records without benchmark fields are `absent`. Only `ok` records with `benchmark_model == 2` are current.

**Basis preprocessing** — shared by both variants: SPQR (column-pivoted QR on $A$) selects a basis $B$ of size $m$ and identifies the non-basic columns $N$. If $A$ is rank-deficient, a secondary SPQR on $A^\top$ identifies dependent rows. A row is dropped only after an LSMR dependence certificate is checked against both $A$ and $b$; a mismatch is `inconsistent_rows`. The same kept-row selection is applied to $A$ and $b$. A sparse LU factorisation of $A_B$ is then computed once and reused by both variants.

**Condition estimation** — both condition numbers are computed matrix-free via ARPACK on `LinearOperator` objects and are **lower bounds** on the true $\kappa$. `svds("LM")` Ritz values underestimate $\sigma_\max$; `svds("SM")` Ritz values overestimate $\sigma_\min$; their ratio is therefore a lower bound on the true condition number. On timeout, random probes provide a lower bound on $\sigma_\max$ or an upper bound on $\sigma_\min$. MNES uses left probes $\|\bar{F}^{\top}u\|$ for the latter; OSS uses $\|Mw\|$. These directions preserve the lower-bound guarantee for $\kappa$. The implementation uses floating-point numerical evidence rather than certified arithmetic, so roundoff and solver behavior remain practical caveats.

**Sparsity estimation** — model 2 forms sampled rows and columns of the actual MNES or OSS operator through sparse products and the selected-basis LU. It records the largest numerical nnz count observed, with entries below $10^{-12}$ times the largest magnitude in that vector omitted. Blocks of at most 64 indices are enumerated and recorded as `exact`; larger blocks use 64 distinct indices from a seeded generator and are recorded as `sampled`. Sampling a maximum and thresholding can only undercount the numerically formed maximum. The modeled and floor lines increase with $s$, while the best-known line is independent of $s$, so underestimating $s$ never overstates a reported quantity. This is also floating-point evidence, not certified arithmetic. The method is stored in `sparsity_method_v`.

#### MNES — `mnes`

The reduced matrix $\bar{F} = A_B^{-1} A_N \in \mathbb{R}^{m \times (n-m)}$ is wrapped as a `LinearOperator` (matvec: $v \mapsto A_B^{-1}(A_N v)$). Since $\lambda_i(\hat{M}) = 1 + \sigma_i(\bar{F})^2$:

$$\kappa(\hat{M}) = \frac{1 + \sigma_\max(\bar{F})^2}{1 + \sigma_\min(\bar{F})^2}.$$

$\sigma_\max$ is computed via `svds("LM")`. When $n - m < m$, the rank of $\bar{F}$ is at most $n - m < m$, so $\bar{F}\bar{F}^\top$ has a null space and $\lambda_\min(\hat{M}) = 1$ exactly — the second `svds` call is skipped. Otherwise $\sigma_\min$ is found via `svds("SM")` with the timeout/probe fallback. For sparsity, a sampled row is formed as $e_i + \bar F(\bar F^\top e_i)$. If $A_N$ is empty or has no nonzeros, $\hat M=I$ and $s=1$ exactly.

#### OSS — `oss`

The null-space basis $V \in \mathbb{R}^{n \times (n-m)}$ is defined implicitly by $V_B = -A_B^{-1} A_N$, $V_N = I_{n-m}$, and $M = [-A^\top \mid V]$ (at $x = s = \mathbf{1}$) is wrapped as a `LinearOperator`. The condition number is $\kappa(M) = \sigma_\max(M) / \sigma_\min(M)$, with $\sigma_\max$ via `svds("LM")` and $\sigma_\min$ via `svds("SM")` with the timeout/probe fallback. Sparsity sampling covers both row blocks and both column blocks of this actual $n\times n$ operator. In particular, it forms $V$ entries through $A_B^{-1}A_N$ rather than assuming that those entries are dense.

**Screening estimate hierarchy** — this benchmark fixes the MNES/OSS formulation, the SPQR basis, the $x=s=\mathbf 1$ initialization, a QLSA and readout model, and a serial one-cycle-per-query oracle. It does not claim an instance-wise bound over all QIPMs or QLSAs. For each variant it reports three per-solve query lines:

1. Modeled Chebyshev pipeline: `cycle_count_qlsa(s, κ, ε_qlsa)`, using the existing Lefterovici et al. formula.
2. Best-known solver estimate: $\lceil\kappa\ln(2\sqrt2/\varepsilon_\mathrm{qlsa})\rceil$, following Dalzell's known-norm kernel-reflection shortcut and the constant-factor analysis of Costa et al. Using target-matrix $\kappa$ in place of block-encoding $\kappa_\mathrm{BE}\geq\kappa$ is quantum-favorable.
3. Scaling-only floor: $\lceil\kappa\sqrt{s}\rceil$, representing the Mori et al. worst-case $\Omega(\kappa\sqrt{s})$ scaling with its unknown constant set to 1. It is not a proven per-instance floor.

All three use the shared state-preparation-unitary readout estimate

$$R=\left\lceil\frac{d-1}{\varepsilon_\mathrm{tomo}}\right\rceil,\qquad d=m\ \text{(MNES)},\quad d=n\ \text{(OSS)}.$$

This follows the $\widetilde\Theta(d/\varepsilon)$ pure-state $\ell_2$ tomography result of van Apeldoorn et al. (SODA 2023, Theorems 23 and 52). It is treated as at or below every known readout protocol, not as a theorem-exact bound because constants and polylogarithms are hidden. Both $\varepsilon_\mathrm{qlsa}$ and $\varepsilon_\mathrm{tomo}$ are 0.1 per solve, justified by iterative refinement following Mohammadisiahroudi et al. The two errors add by the triangle inequality; the screen deliberately does not charge tighter component precision. Each total is its query line times $R$.

Model-1 ledgers cannot be migrated because their stored sparsity and repetition bases are stale. The former `--refresh-counts` command has therefore been removed. Re-run the benchmark stage; `--clear` remains available. Because `benchmark_model` is record-wide, upgrading a model-1 ledger discards both variants' stale benchmark fields even when the rerun selects only one variant.

### 5. Plot

`plot.py` produces advantage curves, fixed-cycle ratio plots, and difficulty histograms from `.data`. The default classical baseline is now `--solver highs-std`; `glpk` and `highs-mps` remain available.

Use `python plot.py --ratio` for fixed-cycle-time quantum/classical ratios. The default box style shows modeled serial query work $Q\times R$ and one QLSA state preparation (no readout) for MNES and OSS; `--ratio-style ecdf` selects empirical CDFs. $Q\times R$ is serial query work, not a hardware-independent wall-clock bound. `--cycle-time` sets an illustrative optimistic proxy for a logical cycle and defaults to $8\times10^{-10}$ s (800 ps), the $\sqrt{\mathrm{SWAP}}$ two-qubit gate reported by He et al., *Nature* **571**, 371 (2019), which Sec. V of the paper uses as an optimistic physical proxy. Plots report records with stale benchmark data (benchmark fields present but not model 2) and silently skip records that were never benchmarked; ratio plots also report invalid breakdowns, non-positive classical runtimes, and ratios too large for plotting.

Plotting enables Matplotlib's LaTeX renderer, so a working LaTeX installation is required in addition to the Python packages.

The cache pipeline assumes a single writer per instance; do not run stages concurrently against the same instance directory.

## Installation

```bash
# System dependency (macOS)
brew install suite-sparse

pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `highspy`, `sparseqr`, `matplotlib`, `tqdm`.

## Tests

```bash
python -m pytest tests/
```
