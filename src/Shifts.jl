# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
This module groups all pre-defined shift strategies.

* [`Shifts.Heuristic`](@ref)
* [`Shifts.Projection`](@ref)
* [`Shifts.Cyclic`](@ref)
* [`Shifts.Wrapped`](@ref)

# Extended help

To define a custom shift strategy,
create a mutable subtype of `Shifts.Strategy`,
define a method for [`Shifts.init`](@ref) and, optionally,
methods for [`Shifts.update!`](@ref) and [`Shifts.take!`](@ref).

```julia
struct FortyTwo <: Shifts.Strategy end

Shifts.init(::FortyTwo, _...) = FortyTwo()
Shifts.take!(::FortyTwo) = 42
```

If it is customary to generate multiple shift parameters at once,
that are then to be used one-by-one, define a method for
[`Shifts.take_many!`](@ref) and have [`Shifts.init`](@ref) return a
[`Shifts.BufferedIterator`](@ref).

```julia
struct FibonacciShifts <: Shifts.Strategy
    "Number of shifts to generate at a time"
    n::Int

    function FibonacciShifts(n::Int)
        n >= 2 || error("batch size is too small")
        new(n)
    end
end

struct FibonacciShiftsIterator
    n::Int
    f1::Int
    f2::Int
end

function Shifts.init(f::FibonacciShifts)
    Shifts.BufferedIterator(FibonacciShiftsIterator(f.n, 0, 1))
end

function Shifts.take_many!(it::FibonacciShiftsIterator)
    n = it.n

    # Generate n shifts at once:
    f = Vector{Int}(undef, n)
    f[1] = it.f1
    f[2] = it.f2
    for i in 3:n
        f[i] = f[i-1] + f[i-2]
    end

    # Prepare next batch:
    it.f1 = f[end-1] + f[end]
    it.f2 = f[end] + it.f1
    return f
end
```
"""
module Shifts

export Cyclic, Wrapped
export Heuristic, Projection

abstract type Strategy end

"""
    Shifts.init(::Shifts.Strategy, prob)

Create and initialize a shift generator from problem data.
The returned iterator will immediately be [`Shifts.update!`](@ref)ed
with initial guess and residual of the iteration.
"""
init

"""
    Shifts.update!(shifts, X, R, Vs...)

Pass most recent solution update to shift generator `shifts`.

* `X`: current solution candidate
* `R`: outer factor of residual corresponding to `X`
* `Vs`: outer factors of most recent updates comprising `X`

This operation must be cheap.
Defer the computation of new shift parameters to [`Shifts.take!`](@ref) or [`Shifts.take_many!`](@ref).

Default: no-op.
"""
update!(_, _, _, _...) = nothing

"""
    Shifts.take!(shifts)

Return the next shift parameter from shift generator `shifts`.

This operation may be expensive.
Compute new shift parameters, if needed.

Default: `popfirst!(shifts)`
"""
take!(shifts) = popfirst!(shifts)

using ..Stuff
include("shifts/helpers.jl")
include("shifts/heuristic.jl")
include("shifts/projection.jl")

end
