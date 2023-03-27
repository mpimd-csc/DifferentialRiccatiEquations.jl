# DifferentialRiccatiEquations.jl

[![Build Status](https://gitlab.mpi-magdeburg.mpg.de/jschulze/DifferentialRiccatiEquations.jl/badges/master/pipeline.svg)](https://gitlab.mpi-magdeburg.mpg.de/jschulze/DifferentialRiccatiEquations.jl/pipelines)
[![Coverage](https://gitlab.mpi-magdeburg.mpg.de/jschulze/DifferentialRiccatiEquations.jl/badges/master/coverage.svg)](https://gitlab.mpi-magdeburg.mpg.de/jschulze/DifferentialRiccatiEquations.jl/commits/master)

This package provides algorithms to solve autonomous Generalized Differential Riccati Equations (GDRE)

```math
\left\{
\begin{aligned}
E^T \dot X E &= C^T C + A^T X E + E^T X A - E^T X BB^T X E,\\
X(t_0) &= X_0.
\end{aligned}
\right.
```

More specifically:

* Dense Rosenbrock methods of orders 1 to 4
* Low-rank symmetric indefinite (LRSIF) Rosenbrock methods of order 1 and 2, $X = LDL^T$

In the latter case, the (generalized) Lyapunov equations arizing in the Rosenbrock stages
are solved using a LRSIF formulation of the Alternating-Direction Implicit (ADI) method,
as described by [LangEtAl2015].
The ADI uses the self-generating parameters described by [Kuerschner2016].

> **Warning**
> The low-rank 2nd order Rosenbrock method suffers from the same problems as described by [LangEtAl2015].

[Kuerschner2016]: https://hdl.handle.net/11858/00-001M-0000-0029-CE18-2
[LangEtAl2015]: https://doi.org/10.1016/j.laa.2015.04.006

The user interface is modeled after [CommonSolve.jl] by providing the `GDREProblem` problem type
as well as the `Ros1`, `Ros2`, `Ros3`, and `Ros4` solver types.

[CommonSolve.jl]: https://github.com/SciML/CommonSolve.jl

# Demos
## Dense formulation

The easiest setting is perhaps the dense one,
i.e. the system matrices `E`, `A`, `B`, and `C`
as well as the solution trajectory `X` are dense.
First, load the system matrices from e.g. `test/Rail371.mat`
and define the problem parameters.

```julia
using DifferentialRiccatiEquations
using LinearAlgebra
using MAT, UnPack

P = matread("Rail371.mat")
@unpack E, A, B, C, X0 = P

tspan = (4500., 0.) # backwards in time
```

Then, instantiate the GDRE and call `solve` on it.

```julia
prob = GDREProblem(E, A, B, C, X0, tspan)
sol = solve(prob, Ros1(); dt=-100)
```

The trajectories $X(t)$, $K(t) := B^T X(t) E$, and $t$ may be accessed as follows.

```julia
sol.X # X(t)
sol.K # K(t) := B^T X(t) E
sol.t # discretization points
```

By default, the state $X$ is only stored at the boundaries of the time span `tspan`,
as one is mostly interested only in the feedback matrices $K$.
To store the full state trajectory, pass `save_state=true` to `solve`.

```julia
sol_full = solve(prob, Ros1(); dt=-100, save_state=true)
```

## Low-rank formulation

Continuing from the dense setup,
assemble a low-rank variant of the initial value,
$X_0 = LDL^T$ where $E^T X_0 E = C^T C / 100$ in this case.
Both dense and sparse factors are allowed for $D$.

```julia
using SparseArrays

q = size(C, 1)
L = E \ C'
D = sparse(0.01I(q))
X0_lr = LDLᵀ(L, D)

Matrix(X0_lr) ≈ X0
```

Passing this low-rank initial value to the GDRE instance
selects the low-rank algorithms and computes the whole trajectories in $X$ that way.
Recall that these trajectories are only stored iff one passes the keyword argument `save_state=true` to `solve`.

```julia
prob_lr = GDREProblem(E, A, B, C, X0_lr, tspan)
sol_lr = solve(prob_lr, Ros1(); dt=-100)
```

> **Note**
> The type of the initial value, `X0` or `X0_lr`,
> dictates the type used for the whole trajectory, `sol.X` and `sol_lr.X`.

