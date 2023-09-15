# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
This module groups all pre-defined shift strategies.
To define a custom one, create a mutable subtype of `ShiftIterator` having a `shifts` field,
and optionally define methods for [`update_shifts!`](@ref) and [`compute_next_shifts!`](@ref),
or [`get_next_shift!`](@ref).

```julia
mutable FortyTwo <: ShiftIterator
    shifts
    MyShifts() = new(Iterators.repeated(-42))
end
```
"""
module Shifts

export update_shifts!, get_next_shift!

abstract type ShiftIterator end

"""
    update_shifts!(::ShiftIterator, X, R, Vs...)

* `X`: current solution candidate
* `R`: outer factor of residual corresponding to `X`
* `Vs`: outer factors of most recent updates comprising `X`
"""
update_shifts!(::ShiftIterator, _, _, _...) = nothing

"""
    get_next_shift!(it::ShiftIterator)

Return the next shift parameter from `it.shifts`, if the latter is not empty.
If it is empty, call [`compute_next_shifts!(it)`](@ref) beforehand, and write the result to `it.shifts`.
"""
function get_next_shift!(it::ShiftIterator)
    if isempty(it.shifts)
        it.shifts = compute_next_shifts!(it)
        @debug "Obtained $(length(it.shifts)) new shifts"
    end
    pop!(it.shifts)
end

"""
    compute_next_shifts!(::ShiftIterator)

Returns the next shift parameters.
This may be an expensive operation.
"""
function compute_next_shifts! end

using ..Stuff
include("shifts/kuerschner.jl")

end