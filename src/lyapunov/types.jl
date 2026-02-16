# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
Generalized algebraic Lyapunov equation

    A'XE + E'XA = -C

having the fields `A`, `E` and `C`.
"""
struct GALEProblem{T}
    E
    A
    C::T

    GALEProblem(E, A, C::T) where {T} = new{T}(E, A, C)
end

function Adapt.adapt_structure(to, prob::GALEProblem)
    E = adapt(to, prob.E)
    A = adapt(to, prob.A)
    C = adapt(to, prob.C)
    return GALEProblem(E, A, C)
end

abstract type LyapunovSolver end

"""
    ADI(inner_alg; kwargs...)
    ADI(; kwargs...)

Alternating Direction-Implicit method to solve algebraic Lyapunov equations.

    solve(prob::GALEProblem, ADI(; kwargs...); solve_kwargs...)

Supported keyword arguments (`kwargs`):

- `inner_alg = Backslash()`:
  algorithm to solve inner block-linear equations arizing at every ADI step
- `maxiters = 100`: maximum number of ADI steps
- `reltol = nothing`:
  relative tolerance of Lyapunov residual; if set to `nothing`, equivalent to `size(prob.A, 1) * eps()`
- `abstol = nothing`:
  absolute tolerance of Lyapunov residual; if set to `nothing`, equivalent to `reltol * norm(prob.C)` (residual of zero value)
- `shifts = Shifts.Projection(2)`:
  shift strategy (oracle) to use; see [`Shifts`](@ref) for more info
- `ignore_initial_guess = false`:
  whether to ignore `initial_guess` passed via `solve_kwargs`
- `compression_interval = 10`:
  compress solution roughly every `compression_interval` iterations
- `compression = true`: disable compression entirely if `false`
- `warn_convergence = true`: warn if method failed to converge
- `inexact = true`:
  whether to allow (more) inexact block-linear solutions following KurF20
- `inexact_forcing = simple_adi_forcing`:
  which relaxation strategy to use; see [`simple_adi_forcing`](@ref) for more info

!!! warning "Experimental"
    The `inexact*` arguments are not part of the stability promises of semantic versioning.
    That is, the API may change or the functionality might be removed any time.
"""
@kwdef struct ADI <: LyapunovSolver
    maxiters::Int = 100
    reltol::Union{Nothing,Real} = nothing
    abstol::Union{Nothing,Real} = nothing
    shifts::Shifts.Strategy = Shifts.Projection(2)
    ignore_initial_guess::Bool = false # use zero if true
    inner_alg::BlockLinearSolver = Backslash() # to solve BlockLinearProblem
    compression_interval::Int = 10 # compress roughly every n iterations
    compression::Bool = true # disable compression entirely if false
    warn_convergence::Bool = true # warn if method failed to converge
    # Keep inexact_* parameter names in sync with Newton:
    inexact::Bool = false
    inexact_forcing = simple_adi_forcing
    # TODO: inexact_hybrid::Bool = true # switch to standard ADI if inexact threshold is too small
    # TODO: inexact_reltol_min::Union{Nothing,Real} = nothing
    # TODO: inexact_reltol_max::Union{Nothing,Real} = nothing
    # TODO: Can I do inexact::Union{Nothing,InexactMethod} that I use for both Newton and ADI?
end

ADI(inner_alg; kwargs...) = ADI(; inner_alg, kwargs...)

function Base.hash(alg::ADI, h::UInt)
    hv = hash(42)
    for p in propertynames(alg)
        hv ⊻= hash(p, hash(getproperty(alg, p)))
    end
    hash(hv, h)
end

struct BartelsStewart <: LyapunovSolver end

@kwdef struct GMRES <: LyapunovSolver
    maxiters::Int = 3 # per restart
    maxrestarts::Int = 0
    reltol::Union{Nothing,Real} = nothing
    abstol::Union{Nothing,Real} = nothing # TODO: Should I remove this in the next release?
    ignore_initial_guess::Bool = false # use zero if true
    compression::Bool = true # disable compression entirely if false
    preconditioner::Union{Nothing,LyapunovSolver} = nothing
end

struct Kronecker <: LyapunovSolver end
