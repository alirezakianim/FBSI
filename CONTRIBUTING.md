# Contributing

Thank you for your interest in FBSI.

- **Questions and bug reports:** open an issue. For bugs, include a minimal example: the cell
  geometry, particle position and radius, and the result you expected.
- **Implementations in other solvers** (OpenFOAM, Fluent UDF, MFIX, LIGGGHTS/CFDEM, in-house codes):
  issues describing your experience or linking to your implementation are very welcome.
- **Pull requests:** keep changes small and readable, since this repository is primarily for
  explaining the method. Make sure both test suites pass:

  ```bash
  pytest python/tests
  cmake -S cpp -B build && cmake --build build && ctest --test-dir build
  ```
