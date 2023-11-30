# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
This module groups all pre-defined shift strategies.
To define a custom one, create a mutable subtype of `Shifts.Strategy`,
define a method for [`Shifts.init`](@ref) and, optionally,
methods for [`Shifts.update!`](@ref) and [`Shifts.take!`](@ref).

```julia
struct FortyTwo <: Shifts.Strategy end

Shifts.init(::FortyTwo, _...) = FortyTwo()
Shifts.take!(::FortyTwo) = 42
```
"""
module Shifts

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
This operation must be cheap.

* `X`: current solution candidate
* `R`: outer factor of residual corresponding to `X`
* `Vs`: outer factors of most recent updates comprising `X`

Default: no-op.
"""
update!(_, _, _, _...) = nothing

"""
    Shifts.take!(shifts)

Return the next shift parameter from shift generator `shifts`.
This operation may be expensive.

Default: `popfirst!(shifts)`
"""
take!(shifts) = popfirst!(shifts)

#
# Some iteration helpers
#

mutable struct BufferedIterator
    shifts::Vector{ComplexF64}
    generator

    BufferedIterator(gen) = new(ComplexF64[], gen)
end

update!(it::BufferedIterator, args...) = update!(it.generator, args...)

take_many!(it) = Tuple(take!(it))

function take!(it::BufferedIterator)
    if isempty(it.shifts)
        it.shifts = take_many!(it.generator)
        @debug "Obtained $(length(it.shifts)) new shifts"
    end
    pop!(it.shifts)
end

using ..Stuff
include("shifts/kuerschner.jl")
include("shifts/penzl.jl")

end
