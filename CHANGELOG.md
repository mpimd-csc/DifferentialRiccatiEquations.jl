# v0.5.x

This release completely changes the user-facing API.
Refer to the docstrings of the algorithm types for more details.

* Breaking: Require Julia v1.10
* Breaking: Remove `test/Rail371.mat`; use `SteelProfile(371)` from MORWiki.jl instead
* Breaking: New default value `ADI(; ignore_initial_guess=false)` differs from v0.4 behavior
* Breaking: Refactor `NewtonADI()` to `Newton(ADI())` (b3314ed882b1afcfbf719b9f955d117a531aae8b)
* Add rudimentary GPU support (a55b117c469a8dc9006d774d76ab733119a3fdc8, ef9c8fe47378a98f3b0c7b2010478f99c92f9930)
* Add optional argument to `Ros1` and `Ros2` to configure ALE solver, e.g., `Ros1(ADI(; maxiters=10))` (73a12bc02d441b3885ba204325b41f466c637c6a)
* Fix `Shifts` helpers: allow, e.g., `Cyclic(Wrapped(real, Heuristic(3, 3, 3)))` (e64da1e1beeae478fe4a5bae2bd76d31bededf06)
* `residual(prob, val)` now allocates in all cases, making its return value safe to modify in-place by the caller

# v0.4.1

* Fix ADI for ALEs with non-symmetric matrices (e62cb82b2c3c3dbf130b2918e2f0a43ed7616133)
* Add `BartelsStewart` and (naive) `Kronecker` solvers for ALEs (22db62a4646c9a422be2d2bc0eea7ace70de1257)
* Export `residual` (8a02c62d98904035186b88e9d7c5a43d103a4685)

# v0.4

* Fix LDLᵀ compression for indefinite objects (d6a649ad62ab6d413ce82e5a2b0090de813a33de)
* Add callbacks to allow user to gather information during `solve` calls;
  see docstring of the `Callbacks` module for more info
* Add configurable shift strategies;
  see docstring of the `Shifts` module for more info
* Change default shift strategy.
  While this is not API breaking, it does affect the convergence behavior.
  (7b32660af73c23c0d710b215705688842aa0bb70)
* Fix order of automatic/projection shifts: ensure that complex shifts occur in
  conjugated pairs directly one after the other (e36a1163f9db4796b334fbdf23c23ea4fd0aab9d)
* Add Inexact Newton method following Dembo et al. (1982) and Benner et al. (2015) to solve AREs;
  see docstring of `NewtonADI` for more info

# v0.3

* Add license
* Improve documentation
* Breaking: Rename keyword arguments of `solve(::GALEProblem, ::ADI; nsteps, rtol)` to `maxiters` and `reltol`
* Rename default branch to `main`

# v0.2.2

* Fix ADI (425d4001112fcff88b30c58f020b106e10a7ef7b)

# v0.2.1

* Add low-rank Ros2 (1345647c610c4561e0d63e8fbee65a85693d8156)

# v0.2

* Add LDLᵀ factorization (4811939893a98b6ebc6e442f6a85ff0dcde4b42e)
* Add LowRankUpdate representation which supports `\` via Sherman-Morrison-Woodbury (2b00c7bf0d817973d41d883773a68db173faaaa6)
* Add low-rank Ros1 (implicit Euler) (c1d4bcf5c22fb71f85512e78c0071e58ffaf1397)
* Dense solvers now support sparse `E` (331094d0ca4cc84f4ae2d13df41cc52b5d229663)

# v0.1

* Port Rosenbrock solvers from Lang 2017
* Reuse Schur decomposition within Rosenbrock steps (3706742ac179c312b66de1ec41d57a7c2924a7af)
