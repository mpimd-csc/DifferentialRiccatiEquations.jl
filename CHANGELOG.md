# DEV

* Fix LDLᵀ compression for indefinite objects (d6a649ad62ab6d413ce82e5a2b0090de813a33de)
* Add callbacks to allow user to gather information during `solve` calls;
  see docstring of the `Callbacks` module for more info
* Add configurable shift strategies;
  see docstring of the `Shifts` module for more info

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
