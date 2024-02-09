# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
Generalized differential Riccati equation

    E'ẊE = C'C + A'XE + E'XA - E'XBB'XE
    X(t0) = X0

having the fields `E`, `A`, `C`, `X0`, and `tspan`=`(t0, tf)`.
"""
struct GDREProblem{XT}
    E
    A
    B
    C
    X0::XT
    tspan

    GDREProblem(E, A, B, C, X0::XT, tspan) where {XT} = new{XT}(E, A, B, C, X0, tspan)
end

"""
Solution to a generalized differential Riccati equation (DRE)
as returned by [`solve(::GDREProblem, alg; kwargs...)`](@ref GDREProblem).
The solution has three fields:

* `X::Vector{T}`: state `X(t)`; `T` may be a `Matrix` or [`LDLᵀ`](@ref)
* `K::Vector{<:Matrix}`: feedback `K(t) := B' * X(t) * E`
* `t::Vector{<:Real}`: discretization time

By default, the state `X` is only stored at the boundaries of the time span,
as one is mostly interested only in the feedback matrices `K`.
To store the full state trajectory, pass `save_state=true` to `solve`.
"""
struct DRESolution
    X
    K
    t
end

"""
Generalized algebraic (continuous time) algebraic Riccati equation

    Q + A'XE + E'XA - E'XGXE = 0
"""
struct GAREProblem{TG,TQ}
    E
    A
    G::TG
    Q::TQ
end

abstract type AlgebraicRiccatiSolver end

"""
    NewtonADI()

Kleinman-Newton method to solve algebraic Riccati equations.
The algebraic Lyapunov equations arizing at every Newton steps are solved using the [`ADI`](@ref).

    solve(prob::GAREProblem, NewtonADI(); kwargs...)

Supported keyword arguments:

* `reltol = size(prob.A, 1) * eps()`: relative Riccati residual tolerance
* `maxiters = 5`: maximum number of Newton steps
* `observer`: see [`Callbacks`](@ref)
* `adi_initprev = false`: whether to use previous Newton iterate
  as the initial guess for the [`ADI`](@ref).
  If `false`, the default initial value of zero is used.
* `adi_kwargs::NamedTuple`:
  keyword arguments to pass to `solve(_, ::ADI; adi_kwargs...)`
* `inexact = true`:
  whether to allow (more) inexact Lyapunov solutions
* `inexact_forcing = quadratic_forcing`:
  compute the forcing parameter `η = inexact_forcing(i, residual_norm)`
  as described by Dembo et al. (1982), where
  `i::Int` is the Newton step and
  `residual_norm::Float64` is the norm of the Riccati residual.
  See [`quadratic_forcing`](@ref), and [`superlinear_forcing`](@ref).
* `inexact_hybrid = true`:
  whether to switch to the classical Newton method,
  if the absolute Lyapunov tolerance of the classical Newton method
  is less strict (i.e. larger) than the tolerance `η * residual_norm`.
* `linesearch = true`: whether to perform an Armijo line search
  if the Riccati residual did not decrease sufficiently,
  see e.g. Benner et al. (2015).

Default arguments to Lyapunov solver, which can all be overwritten by `adi_kwargs`:

* `maxiters = 100`: maximum number of ADI iterations
* `observer = observer`
* `initial_guess`: see `adi_initprev` above
* `reltol`: defaults a fraction of the Riccati tolerance, `reltol/10`
* `abstol`: controlled by `inexact*` above, if `inexact = true`.

References:

* Dembo, Eisenstat, Steihaug: Inexact Newton Methods. 1982.
  https://doi.org/10.1137/0719025
* Benner, Heinkenschloss, Saak, Weichelt: Inexact low-rank Newton-ADI method for large-scale algebraic Riccati equations. 2015.
  http://www.mpi-magdeburg.mpg.de/preprints/
"""
struct NewtonADI <: AlgebraicRiccatiSolver end
