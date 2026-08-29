# Transform test fixtures

The tests retain only the MPS inputs they execute:

- `equality.mps` exercises the `reduced_to_empty` path.
- `surviving_mixed.mps` and `surviving_range.mps` survive presolve and support objective-oracle and end-to-end checks.

Pure-function tests cover every column and row conversion branch. Handcrafted test archives cover the `.std` NPZ layout, so no golden `.std` fixtures are needed.
