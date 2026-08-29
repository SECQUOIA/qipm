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

Benchmarks quantum cycle counts for solving linear programs (LPs) of the form

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

The transform stage writes `transform_status` (`ok`, `reduced_to_empty`, `infeasible`, `unbounded`, `unbounded_or_infeasible`, or `error:<ExceptionName>`) to `.data`. An unchanged successful transform preserves downstream results. A changed successful transform clears them; every non-success conversion outcome removes a stale `.std` and clears them.

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

Each solve runs in a subprocess with a 10-minute timeout. In `both` mode, if the `.mps` solve times out, the `.std` solve is skipped for that instance. Wall-clock solve times are written to the instance's `.data` JSON and serve as the classical baseline for the quantum advantage comparison.

`solve_status_mps` and `solve_status_std` record `ok`, `ok_ipm`, `timeout`, `crashed`, `non_optimal`, or `error:<ExceptionName>`. Runtime keys are present only for optimal solves.

### 4. Benchmark

Compute QLSA cycle counts and write to `.data`:

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

For each instance, the script reads $A$ from the `.std` file and writes three keys per variant into the instance's `.data` JSON: the cycle count (`cycle_count_mnes` / `cycle_count_oss`), the sparsity parameter $s$ (`sparsity_mnes` / `sparsity_oss`), and the condition number $\kappa$ (`cond_mnes` / `cond_oss`).

`status_mnes` and `status_oss` record `ok`, `timeout`, `crashed`, `skipped_too_large`, `skipped_degenerate`, `rank_uncertain`, or `error:<ExceptionName>`. Instances with more than 100,000 rows are recorded as `skipped_too_large`.

**Basis preprocessing** — shared by both variants: SPQR (column-pivoted QR on $A$) selects a basis $B$ of size $m$ and identifies the non-basic columns $N$. If $A$ is rank-deficient, a secondary SPQR on $A^\top$ drops redundant rows. A sparse LU factorisation of $A_B$ is then computed once and reused by both variants for all subsequent triangular solves.

**Condition estimation** — both condition numbers are computed matrix-free via ARPACK on `LinearOperator` objects and are **lower bounds** on the true $\kappa$. `svds("LM")` Ritz values underestimate $\sigma_\max$; `svds("SM")` Ritz values overestimate $\sigma_\min$; their ratio is therefore a lower bound on the true condition number. On timeout, random probes provide a lower bound on $\sigma_\max$ or an upper bound on $\sigma_\min$. MNES uses left probes $\|\bar{F}^{\top}u\|$ for the latter; OSS uses $\|Mw\|$. These directions preserve the lower-bound guarantee for $\kappa$.

#### MNES — `mnes`

The reduced matrix $\bar{F} = A_B^{-1} A_N \in \mathbb{R}^{m \times (n-m)}$ is wrapped as a `LinearOperator` (matvec: $v \mapsto A_B^{-1}(A_N v)$). Since $\lambda_i(\hat{M}) = 1 + \sigma_i(\bar{F})^2$:

$$\kappa(\hat{M}) = \frac{1 + \sigma_\max(\bar{F})^2}{1 + \sigma_\min(\bar{F})^2}.$$

$\sigma_\max$ is computed via `svds("LM")`. When $n - m < m$, the rank of $\bar{F}$ is at most $n - m < m$, so $\bar{F}\bar{F}^\top$ has a null space and $\lambda_\min(\hat{M}) = 1$ exactly — the second `svds` call is skipped. Otherwise $\sigma_\min$ is found via `svds("SM")` with the timeout/probe fallback. The QLSA sparsity parameter is $s = m$ since $\hat{M}$ is generically dense $m \times m$.

#### OSS — `oss`

The null-space basis $V \in \mathbb{R}^{n \times (n-m)}$ is defined implicitly by $V_B = -A_B^{-1} A_N$, $V_N = I_{n-m}$, and $M = [-A^\top \mid V]$ (at $x = s = \mathbf{1}$) is wrapped as a `LinearOperator`. The condition number is $\kappa(M) = \sigma_\max(M) / \sigma_\min(M)$, with $\sigma_\max$ via `svds("LM")` and $\sigma_\min$ via `svds("SM")` with the timeout/probe fallback. The QLSA sparsity parameter $s$ is the maximum nnz over all rows and columns of $M$:

$$s = \max\!\bigl(\underbrace{\text{max row-nnz}(A)}_{\text{z}_y\text{ columns}},\ \underbrace{m+1}_{\text{z}_\lambda\text{ columns}},\ \underbrace{\max_{i\in B}\text{col-nnz}_i(A)+n_N}_{\text{B-rows}}\bigr).$$

The $\mathrm{z}_y$ columns equal the columns of $-A^\top$ (nnz of column $j$ = nnz of row $j$ of $A$). The $\mathrm{z}_\lambda$ columns each have $m$ nonzeros in the $B$-rows (from the dense $A_B^{-1}A_N$ column) plus one in the $N$-rows. The $B$-rows dominate among rows: each $B$-row $i$ has $\mathrm{col\text{-}nnz}_i(A)$ entries from $-A^\top$ plus $n_N$ dense entries from $V_{B,:} = -A_B^{-1}A_N$; $N$-rows have only one nonzero from $V$ and are dominated by the other terms.

When $n_N=0$, MNES uses the exact special case $s=1$; OSS has no $m+1$ null-space-column term and uses the maximum row or column nnz of $A$.

**Cycle count formula** — a single QLSA call costs $\mathcal{Q}=$`cycle_count_qlsa(s, κ, ε)` cycles (Chebyshev query count). The total cycle count is

$$
\text{cycle count} = \mathcal{Q}\times
\begin{cases}
\left\lceil(m-1)/\varepsilon^2\right\rceil & \text{MNES},\\
\left\lceil(2n-1)/\varepsilon^2\right\rceil & \text{OSS}.
\end{cases}
$$

OSS uses the larger factor because of its Hermitian dilation. The required repetition count is rounded up, and $\varepsilon = 0.1$.

### 5. Plot

`plot.py` produces advantage curves and difficulty histograms from `.data`. It enables Matplotlib's LaTeX renderer, so a working LaTeX installation is required in addition to the Python packages.

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
