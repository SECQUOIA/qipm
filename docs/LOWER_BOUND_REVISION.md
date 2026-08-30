# Benchmark model 2: screening-estimate revision

## Why the benchmark model changed

Model 1 combined four increasing factors and called their product a quantum cycle-count lower bound. That interpretation was not defensible. Three factors were estimates in the cost-increasing direction, while the QLSA factor was the cost of one chosen algorithm rather than a universal floor. In addition, the worst-case lower bounds available in the literature are not per-instance statements and cannot generally be multiplied together.

The readout model was internally inconsistent with the rest of the pipeline. Model 1 charged independent copies of the solution state while also assuming deterministic state preparation with no amplitude-amplification repetitions. Deterministic coherent preparation supplies the state-preparation unitary used by the stronger readout model.

| Ingredient | Model-1 choice | Why its direction was wrong for screening |
|---|---|---|
| Tomography | $\lceil(d-1)/\varepsilon^2\rceil$ copy-access cost | State-preparation-unitary pure-state tomography has $\widetilde\Theta(d/\varepsilon)$ cost, so the copy model overcharged readout. |
| OSS readout dimension | $d=2n$ from Hermitian dilation | The solution state occupies only the $n$-coordinate nonzero block. |
| Sparsity | $s=m$ for MNES and structural dense assumptions for OSS | These were upper estimates of a parameter that increases the modeled and floor lines; the best-known line is independent of it. Model 1 also left $s=m$ when $A_N$ had no nonzeros, although $\hat M=I$ means $s=1$; model 2 returns 1 exactly. |
| QLSA | Lefterovici et al. Chebyshev construction presented as a floor | It is a modeled algorithm cost; better solvers are known, and known lower bounds are worst-case asymptotic results. |

Rank reduction also needed a semantic check. Dependence of a row of $A$ does not justify deleting the corresponding equation unless $b$ satisfies the same dependence.

## Old and new formulas

Benchmark model 2 retains the modeled Chebyshev line and adds two clearly labeled comparison lines.

| Quantity | Model 1 | Model 2 |
|---|---|---|
| Prepared-state error | Shared $\varepsilon=0.1$ | $\varepsilon_\mathrm{qlsa}=0.1$ |
| Extracted-vector error | Same shared value | Separate $\varepsilon_\mathrm{tomo}=0.1$ |
| MNES readout | $\lceil(m-1)/\varepsilon^2\rceil$ | $R=\lceil(m-1)/\varepsilon_\mathrm{tomo}\rceil$ |
| OSS readout | $\lceil(2n-1)/\varepsilon^2\rceil$ | $R=\lceil(n-1)/\varepsilon_\mathrm{tomo}\rceil$ |
| Sparsity | Structural upper estimate | Seeded sampled maximum numerical nnz, or exhaustive enumeration for small blocks |
| Modeled queries | Chebyshev $\mathcal Q_\mathrm{Cheb}(s,\kappa,\varepsilon)$ | Same formula, now labeled modeled pipeline |
| Best-known queries | Not reported | $\lceil\kappa\ln(2\sqrt2/\varepsilon_\mathrm{qlsa})\rceil$ |
| Scaling-only floor | Not reported | $\lceil\kappa\sqrt{s}\rceil$, unknown $\Omega$ constant set to 1 |
| Total | One $\mathcal Q R$ value | One total per query line, each using the shared $R$ |

Van Apeldoorn, Cornelissen, Gilyén, and Nannicini give the state-preparation-unitary $\ell_2$ tomography result in SODA 2023 (Theorems 23 and 52). It replaces the copy-access bound associated with Haah et al. Because $\widetilde\Theta$ hides constants and polylogarithms, the implementation treats $\lceil(d-1)/\varepsilon\rceil$ as an estimate at or below every known readout protocol, not as a theorem-exact lower bound.

The modeled line retains the Chebyshev query expression used from Lefterovici et al. Dalzell's kernel-reflection shortcut (arXiv:2406.12086), together with the constant-factor analysis of Costa et al., motivates the best-known line. The code substitutes target-matrix $\kappa$ for the block-encoding value $\kappa_\mathrm{BE}=\alpha\lVert M^{-1}\rVert\geq\kappa$, which favors the quantum estimate. Costa et al.'s discrete adiabatic solver independently establishes the $O(\kappa\log(1/\varepsilon))$ best-known scaling.

Mori, Kikuchi, Benedetti, and Rosenkranz prove the worst-case sparse-access $\Omega(\kappa\sqrt{s})$ scaling used for the third line. Its unknown constant is set to one. The separate $\Omega(\kappa\log(1/\varepsilon))$ result is not used because its proof requires $\varepsilon\leq1/11$, while this benchmark uses 0.1.

Both error targets are 0.1 per solve, following the iterative-refinement justification of Mohammadisiahroudi et al. Prepared-state and extraction errors add by the triangle inequality. Model 2 deliberately does not charge the tighter component errors that a combined 0.1 contract would require; this is quantum-favorable.

The reported repetition, floor, and best-known ceilings are evaluated without upward floating-point rounding. Repetitions use exact rational arithmetic, so at $\varepsilon=0.1$ the result is exactly $10(d-1)$. The $\lceil\kappa\sqrt{s}\rceil$ line uses exact integer square comparisons on the rational value of the stored floating-point $\kappa$; a naive floating-point product can exceed the true ceiling by one. The transcendental scale $\ln(2\sqrt2/\varepsilon)$ in the best-known line is rounded down by one unit in the last place before use. An upward-rounded ceiling would overstate a quantity intended to err low.

## What remains sound

The condition-number machinery is unchanged. Matrix-free extreme-singular-value estimates and their probe fallbacks have the correct direction: they underestimate $\sigma_\max$ or overestimate $\sigma_\min$, producing a lower bound on $\kappa$. Exact shortcuts remain in place. These are numerical floating-point certificates rather than formal certified-arithmetic proofs, so roundoff and solver behavior remain caveats.

The pipeline also retains its benevolent omissions and assumptions: one logical cycle per oracle query, deterministic state preparation without amplitude-amplification overhead, no cost for constructing or refreshing data-access oracles, no fault-tolerance or error-correction overhead, no charge for classical QIPM updates and feasibility checks, and no extra precision charge for combining QLSA and readout error. These choices make the screen favorable to the quantum pipeline.

The sparsity direction is now aligned with that intent. Sparse input is canonicalized at the preprocessing boundary by summing duplicate coordinates and removing explicit zeros, because stored-entry counts on non-canonical CSR could otherwise overstate the sparsity underestimate. MNES samples rows of $I+\bar F\bar F^\top$. OSS samples both row blocks and both column blocks of $[-A^\top\mid V]$, forming $V_B=-A_B^{-1}A_N$ through the LU factorization. Entries no larger than $10^{-12}$ times the sampled vector maximum are omitted. Sampling a maximum and thresholding can only undercount the numerical maximum. The modeled and floor lines increase with $s$, while the best-known line is independent of $s$, so an underestimate never overstates a reported quantity. Blocks of at most 64 indices are enumerated and marked `exact`; larger blocks are marked `sampled`, with the result stored as `sparsity_method_v`. This is floating-point evidence, not certified arithmetic.

## What the numbers mean

For each MNES and OSS record, model 2 reports:

1. A modeled Chebyshev-pipeline estimate.
2. A best-known-solver estimate using an optimistic known-norm constant.
3. A scaling-only worst-case floor with an unproven constant set to one.

None is a rigorous per-instance lower bound on every QIPM or QLSA. The first two are costs of specified algorithm models. The third restates a worst-case asymptotic lower-bound scaling at the measured parameters; it does not prove that the particular instance realizes the hard case. Worst-case bounds for separate stages may be attained by different inputs, so multiplying them does not create an end-to-end per-instance theorem.

The products $Q\times R$ measure serial oracle-query work under a one-cycle-per-query convention. They are not hardware-independent wall-clock bounds. The 800 ps conversion used by plotting is an illustrative optimistic proxy for a logical cycle.

The screened population also excludes instances with more than 100,000 rows. Classical comparison time is the wall time of `Highs.run()` and excludes model construction.

## Operational changes

Every record touched by the benchmark stage receives `benchmark_model: 2`. Status summaries and plots accept only model-2 records. An older successful record or a statusless record carrying a finite legacy count is reported as `outdated_model`; a record without benchmark fields is `absent`. Plots report the outdated benchmark records they skip.

Model-1 sparsity and repetition values cannot be repaired arithmetically. A full benchmark rerun is required. The former `benchmark.py --refresh-counts` command and its tests were removed; `benchmark.py --clear` remains available. Because the model marker is record-wide, upgrading a model-1 ledger discards both variants' stale benchmark fields even during a single-variant run.

This change set also corrected the author metadata for the van Apeldoorn et al. tomography entry in the local literature knowledge base.

When SPQR proposes dropping a dependent row, preprocessing accepts the $b$-side check when its residual is at most $10^{-6}\cdot\max(\lVert b\rVert_\infty,1)\cdot(1+\lVert w\rVert_1)$, where $w$ is the LSMR dependence certificate. The $(1+\lVert w\rVert_1)$ factor matches the error amplification permitted by the relative $A$-side gate. This factor is necessarily two-sided: for a very large certificate, a genuine inconsistency of order $\lVert w\rVert_1\lVert b\rVert_\infty 10^{-6}$ is indistinguishable from certificate error and the row is still dropped. This limitation is inherent to numerical dependence certification and affects only instances that are simultaneously rank-deficient, ill-conditioned, and near-infeasible. A mismatch records `inconsistent_rows`; a consistent system applies the same row selection to both $A$ and $b$.
