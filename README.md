# DifferentialRiccatiEquations.jl

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

The user interface hooks into [CommonSolve.jl] by providing the `GDREProblem` problem type
as well as the `Ros1`, `Ros2`, `Ros3`, and `Ros4` solver types.

[CommonSolve.jl]: https://github.com/SciML/CommonSolve.jl

# Getting started

The package can be installed from Julia's REPL:

```
pkg> add DifferentialRiccatiEquations
```

To run the following demos, you further need the following packages and standard libraries:

```
pkg> add LinearAlgebra MORWiki SparseArrays UnPack
```

What follows is a slightly more hands-on version of `test/rail.jl`.
Please refer to the latter for missing details.

## Dense formulation

The easiest setting is perhaps the dense one,
i.e. the system matrices `E`, `A`, `B`, and `C`
as well as the solution trajectory `X` are dense.
First, load the system matrices from, e.g., [MOR Wiki]
and define the problem parameters.

```julia
using DifferentialRiccatiEquations
using LinearAlgebra
using MORWiki: SteelProfile, assemble
using UnPack: @unpack

@unpack E, A, B, C = assemble(SteelProfile(371))

# Ensure dense storage:
B = Matrix(B)
C = Matrix(C)

# Assemble initial value:
E⁻¹Cᵀ = E \ Matrix(C')
E⁻¹Cᵀ ./= 10
X0 = E⁻¹Cᵀ * (E⁻¹Cᵀ)'

# Problem parameters:
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

```julia
using SparseArrays

q = size(C, 1)
L = E \ C'
D = Matrix(0.01I(q))
X0_lr = lowrank(L, D)

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

## Solver introspection / Callbacks

To record information during the solution process,
e.g. the residual norms of every ADI step at every GDRE time step,
define a custom observer object and associated callback methods.
Refer to the documentation of the `Callbacks` module for further information.

```
julia> import DifferentialRiccatiEquations.Callbacks

help?> Callbacks
```

Note that there are currently no pre-built observers.

## ADI shift parameter selection

The ADI shifts may be configured using keyword arguments of `ADI`.

```julia
adi = ADI(; shifts = Shifts.Projection(2))
solve(::GALEProblem, adi)
solve(::GDREProblem, Ros1(adi))
solve(::GAREProblem, Newton(adi))
```

Pre-built shift strategies include:

* `Heuristic` shifts described by [Penzl1999]
* `Projection` shifts described by [BennerKuerschnerSaak2014]
* User-supplied shifts via the `Cyclic` wrapper

Refer to the documentation of the `Shifts` module for further information.

```
julia> import DifferentialRiccatiEquations.Shifts

help?> Shifts
```

# Known issues

* ADI on GPU breaks for complex-valued shifts

# Acknowledgments

I would like to thank the code reviewers:

* Jens Saak (https://github.com/drittelhacker)
* Martin Köhler (https://github.com/grisuthedragon)
* Fan Wang (https://github.com/FanWang00)

# License

The DifferentialRiccatiEquations package is licensed under [MIT], see `LICENSE`.

[Penzl1999]: https://doi.org/10.1137/S1064827598347666
[BennerSaak2005]: http://nbn-resolving.de/urn:nbn:de:swb:ch1-200601597
[BennerKuerschnerSaak2014]: https://www.emis.de/journals/ETNA/vol.43.2014-2015/pp142-162.dir/pp142-162.pdf
[CC-BY-4.0]: https://spdx.org/licenses/CC-BY-4.0.html
[MIT]: https://spdx.org/licenses/MIT.html
[MOR Wiki]: http://modelreduction.org/index.php/Steel_Profile
