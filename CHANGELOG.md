# DEV

# v0.3

* Add license
* Improve documentation
* Breaking: Rename keyword arguments of `solve(::GALEProblem, ::ADI; nsteps, rtol)` to `maxiters` and `reltol`
* Rename default branch to `main`

# v0.2.2

* Fix ADI (425d4001112fcff88b30c58f020b106e10a7ef7b)

# v0.2.1

* Add low-rank Ros2 (!4)

# v0.2

* Add LDLᵀ factorization (4811939893a98b6ebc6e442f6a85ff0dcde4b42e)
* Add LowRankUpdate representation which supports `\` via Sherman-Morrison-Woodbury (2b00c7bf0d817973d41d883773a68db173faaaa6)
* Add low-rank Ros1 (implicit Euler) (!1)
* Dense solvers now support sparse `E` (!2)

# v0.1

* Port Rosenbrock solvers from Lang 2017
* Reuse Schur decomposition within Rosenbrock steps (3706742ac179c312b66de1ec41d57a7c2924a7af)
