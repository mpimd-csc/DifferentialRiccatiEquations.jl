# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using LinearAlgebra, SparseArrays
using UnPack
using Compat: keepat!

"""
    Shifts.Projection(u::Int)

Compute shift parameters based on the `u` most recent increments comprising the solution candidate.

Only even `u > 1` are allowed, such that an ADI double-step can properly be accounted for.

See section 5.3.1 of

> Kürschner: Efficient low-rank solution of large-scale matrix equations.
> Otto-von-Guericke-Universität Magdeburg (2016).

The strategy has first been presented in

> Benner, Kürschner, Saak: Self-generating and efficient shift parameters in ADI methods for large Lyapunov and Sylvester equations,
> Electronic Transactions on Numerical Analysis, 43 (2014), pp. 142-162.
> https://etna.math.kent.edu/volumes/2011-2020/vol43/abstract.php?vol=43&pages=142-162
"""
struct Projection <: Strategy
    n_history::Int

    function Projection(u)
        isodd(u) && throw(ArgumentError("History must be even; got $u"))
        new(u)
    end
end

mutable struct ProjectionShiftIterator
    prob
    n_history::Int
    Vs::Vector{Any}
end

function init(strategy::Projection, prob)
    it = ProjectionShiftIterator(prob, strategy.n_history, [])
    BufferedIterator(it)
end

function update!(it::ProjectionShiftIterator, _, R, Vs...)
    isempty(Vs) && push!(it.Vs, R)
    append!(it.Vs, Vs)
    lst = length(it.Vs)
    fst = max(1, lst - it.n_history + 1)
    keepat!(it.Vs, fst:lst)
    return
end

function take_many!(it::ProjectionShiftIterator)
    @unpack E, A = it.prob
    @unpack Vs = it

    N = hcat(Vs...)::AbstractMatrix{<:Real}
    Q = orth(N)
    Ẽ = restrict(E, Q)
    Ã = restrict(A, Q)
    λ = eigvals(Ã, Ẽ)
    # TODO: flip values at imaginary axes instead
    λ₋ = filter(l -> real(l) < 0, λ)

    safe_sort!(λ₋)

    return λ₋
end

