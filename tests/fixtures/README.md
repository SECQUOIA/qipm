# Transform test fixtures (MPS → presolve → standard form)

The legacy small fixtures and reference `.std` files document individual MPS constructs. HiGHS presolves most of them to empty, so they are no longer used as golden pipeline outputs.

| Fixture         | Purpose                                                            |
|-----------------|--------------------------------------------------------------------|
| **min_sum**     | ≤ row (L), two vars with lower bound 0 → slack in standard form    |
| **equality**    | = row (E), two vars ≥ 0 → already standard form                    |
| **three_var**   | Two ≤ rows, three vars → presolve can reduce formulation           |
| **bounded_var** | One variable with both LO and UP (bounded column branch)           |
| **lower_row**   | ≥ row (G) → row “lower only” branch (slack with -1)                |
| **free_var**    | One FR (free) variable → x = x⁺ − x⁻ branch                        |
| **upper_var**   | One variable with MI + UP (upper-only column branch)               |
| **range_row**   | L row + RANGES → row with finite lo < hi (range constraint branch) |
| **surviving_mixed** | Mixed row and bound types that empirically survive HiGHS presolve |
| **surviving_range** | Dense full-row-rank LP that empirically survives HiGHS presolve |

Tests use the two surviving fixtures for objective-oracle and end-to-end checks, and an old fixture for the `reduced_to_empty` path. Pure-function tests cover every column branch (including fixed variables) and row branch directly instead of trusting golden NPZ files.
