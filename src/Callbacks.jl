"""
This module groups all callback functions
that are called at various points during the solution process.
See their respective docstrings for more information.
Every callback takes an observer as its first argument,
which is passed via an optional keyword argument to `solve`.

During `solve(::GALEProblem, ::LyapunovSolver; observer)`:

* [`observe_gale_start!`](@ref)
* [`observe_gale_step!`](@ref)
* [`observe_gale_done!`](@ref)
* [`observe_gale_failed!`](@ref)
* [`observe_gale_metadata!`](@ref)

During `solve(::GAREProblem, ::AlgebraicRiccatiSolver; observer)`:

* [`observe_gare_start!`](@ref)
* [`observe_gare_step!`](@ref)
* [`observe_gare_done!`](@ref)
* [`observe_gare_failed!`](@ref)

During `solve(::GDREProblem, ::Algorithm; observer)`:

* [`observe_gdre_start!`](@ref)
* [`observe_gdre_step!`](@ref)
* [`observe_gdre_done!`](@ref)

# Extended help

Hook into above callbacks by first defining a custom observer type.

```julia
mutable struct ResidualObserver
    norms::Vector{Float64}
    abstol::Float64

    ResidualObserver() = new(Float64[], -1.0)
end
```

Then, create custom methods to above callbacks.
If the observer needs to store any information,
use some global variables (not recommended),
have the observer be mutable.
Note that `Callbacks` has to be imported manually;
this is a deliberate choice.

```julia
import DifferentialRiccatiEquations.Callbacks

function Callbacks.observe_gale_step!(o::ResidualObserver, _prob, _alg, abstol::Float64, _reltol)
    o.abstol = abstol
end

function Callbacks.observe_gale_step!(o::ResidualObserver, _iter, _sol, _residual, residual_norm::Float64)
    push!(o.norms, residual_norm)
end
```

The observer is passed into the solution procedure as follows.

```julia
prob = GALEProblem(E, A, C)
alg = ADI()
obs = ResidualObserver()

solve(prob, alg; observer=obs)

@show obs.norms[end] <= obs.abstol
```

!!! todo
    Update extended help to use doctests.
"""
module Callbacks

export observe_gale_start!,
       observe_gale_step!,
       observe_gale_done!,
       observe_gale_failed!,
       observe_gale_metadata!
export observe_gare_start!,
       observe_gare_step!,
       observe_gare_done!,
       observe_gare_failed!
export observe_gdre_start!,
       observe_gdre_step!,
       observe_gdre_done!

"""
    observe_gale_start!(observer, prob::GALEProblem, alg::LyapunovSolver,
                        abstol, reltol)

Notify `observer` at the start of solving the GALE.
`abstol` and `reltol` denote the absolute and relative residual thresholds,
at which the algorithm considers itself converged.
"""
observe_gale_start!(::Any, args...) = nothing

const COMMON_GALE_DESC = """
The observer may compute ans store any metrics of the subsequent arguments,
but it must not modify any of them.

* `X`: solution candidate
* `residual`: residual corresponding to `X`;
  usually of the same data type as `X`
* `residual_norm`: internal approximation of the norm of `residual`
"""

"""
    observe_gale_step!(observer, iter::Int, X, residual, residual_norm)

Notify `observer` for an iterative GALE algorithm,
that iteration number `iter` has been completed.
$COMMON_GALE_DESC

!!! note
    The iterations `iter` may not be consequtive.
    If an algorithm computes multiple steps at once
    and has no (cheap) representation of the intermediate solution candidates,
    the difference between the values of `iter`
    of subsequent calls to `observe_gale_step!`
    may differ by more than one.
"""
observe_gale_step!(::Any, args...) = nothing

"""
    observe_gale_done!(observer, iters::Int, X, residual, residual_norm)

Notify `observer` at the end of solving the GALE.
$COMMON_GALE_DESC
"""
observe_gale_done!(::Any, args...) = nothing

"""
    observe_gale_failed!(observer)

Notify `observer` that the algorithm has failed to solve the GALE.
[`observe_gale_done!`](@ref) will be called regardless.
"""
observe_gale_failed!(::Any) = nothing

"""
    observe_gale_metadata!(observer, desc::String, metadata)

Notify `observer` on some `metadata` the algorithm has computed.
`desc` gives a brief description of the metadata.

### Example

The [`ADI`](@ref) calls `observe_gale_metadata!(observer, "ADI shifts", μ)`,
where `μ` are the (newly) computed ADI shift parameters.
"""
observe_gale_metadata!(::Any, args...) = nothing

"""
    observe_gdre_start!(observer, ::GDREProblem, ::Algorithm)

Notify `observer` at the start of solving the GDRE.
"""
observe_gdre_start!(::Any, args...) = nothing

"""
    observe_gdre_step!(observer, t::Float64, X, K)

Notify `observer` that the step to time point `t` has been completed.

* `X`: solution at time `t`
* `K`: feedback matrix `K = B' * X * E`
  where `B` denotes the input map of the associated [`GDREProblem`](@ref)
"""
observe_gdre_step!(::Any, args...) = nothing


"""
    observe_gdre_done!(observer)

Notify `observer` at the end of solving the GDRE.
"""
observe_gdre_done!(::Any) = nothing

# TODO: refactor callbacks to receive problem instance:
#       observe_start(::Handler, ::Problem, args...)
observe_gare_start!(::Any, args...) = nothing
observe_gare_step!(::Any, args...) = nothing
observe_gare_done!(::Any, args...) = nothing
observe_gare_failed!(::Any) = nothing

end
