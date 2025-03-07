# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Base.Iterators: Stateful, cycle

"""
    Cyclic(::Shifts.Strategy)
    Cyclic(values)

Cycle through precomputed `values` or the shifts produced by the inner strategy.
That is, continue with the first parameter once the last one has been consumed.

Examples:

```julia
Cyclic(Heuristic(10, 20, 20))
Cyclic(Float64[-1, -2, -3])
```
"""
struct Cyclic <: Strategy
    inner
end

"""
    Wrapped(func!, ::Shifts.Strategy)

Apply `func!` to the set of shifts produced by the inner strategy via [`Shifts.take_many!`](@ref).
This strategy may be used, e.g., to filter or reorder the shifts.
Complex-valued shifts must occur in conjugated pairs.

Examples:

```julia
Wrapped(reverse, Projection(2))
Wrapped(Projection(4)) do shifts
    filter(s -> real(s) < -1, shifts)
end
Wrapped(Heuristic(10, 10, 10)) do shifts
    filter(isreal, shifts)
end
```
"""
struct Wrapped <: Strategy
    func!
    inner::Strategy
end

###

"""
    BufferedIterator(generator)

Initialize an internal buffer of type `Vector{<:Number}` from
[`Shifts.take_many!(generator)`](@ref Shifts.take_many!)
and return shifts one-by-one using `popfirst!`.
Refill the buffer once it is depleated.
"""
mutable struct BufferedIterator
    buffer::Vector{<:Number}
    generator

    BufferedIterator(gen) = new(ComplexF64[], gen)
end

"""
    Shifts.take_many!(generator)

Return a `Vector{<:Number}` of shift parameters to be used
within a [`Shifts.BufferedIterator`](@ref).
"""
take_many!

mutable struct WrappedIterator
    func!
    generator
end

# Allow Cyclic(42) for convenience:
_init(values, _) = values
_init(s::Strategy, prob) = init(s, prob)
init(c::Cyclic, prob) = Stateful(cycle(take_many!(_init(c.inner, prob))))

# Ensure that BufferedIterator remains the outer-most structure:
_wrap(it, func!) = WrappedIterator(func!, it)
_wrap(it::BufferedIterator, func!) = BufferedIterator(WrappedIterator(func!, it.generator))
init(w::Wrapped, prob) = _wrap(init(w.inner, prob), w.func!)

update!(it::BufferedIterator, args...) = update!(it.generator, args...)
update!(it::WrappedIterator, args...) = update!(it.generator, args...)

take_many!(values) = values
take_many!(it::WrappedIterator) = it.func!(take_many!(it.generator))

function take!(it::BufferedIterator)
    if isempty(it.buffer)
        it.buffer = take_many!(it.generator)
        @debug "Obtained $(length(it.buffer)) new shifts"
    end
    # TODO: Using `popfirst!` feels inefficient, even though there should be only 10s of elements buffered.
    popfirst!(it.buffer)
end

###

"""
    safe_sort!(shifts)

Ensure that complex conjugated values are located adjacent to one another.
"""
safe_sort!(shifts) = sort!(shifts; by = v -> (real(v), abs(imag(v))))
