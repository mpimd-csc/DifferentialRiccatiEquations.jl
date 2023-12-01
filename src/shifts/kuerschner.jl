# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using LinearAlgebra, SparseArrays
using UnPack
using Compat: keepat!

"""
    Shifts.KuerschnerV(u::Int)

Compute shift parameters based on the `u` most recent increments comprising the solution candidate to `prob`.

It is recommended to use even `u > 1`, such that an ADI double-step can properly be accounted for.

See section 5.3.1 of

> Kürschner: Efficient low-rank solution of large-scale matrix equations.
> Otto-von-Guericke-Universität Magdeburg (2016).
"""
struct KuerschnerV <: Strategy
    n_history::Int
end

mutable struct KuerschnerVIterator
    prob
    n_history::Int
    Vs::Vector{Any}
end

function init(strategy::KuerschnerV, prob)
    it = KuerschnerVIterator(prob, strategy.n_history, [])
    BufferedIterator(it)
end

function update!(it::KuerschnerVIterator, _, R, Vs...)
    isempty(Vs) && push!(it.Vs, R)
    append!(it.Vs, Vs)
    lst = length(it.Vs)
    fst = max(1, lst - it.n_history + 1)
    keepat!(it.Vs, fst:lst)
    return
end

function take_many!(it::KuerschnerVIterator)
    @unpack E, A = it.prob
    @unpack Vs = it

    N = hcat(Vs...)::AbstractMatrix{<:Real}
    Q = orth(N)
    Ẽ = restrict(E, Q)
    Ã = restrict(A, Q)
    λ = eigvals(Ã, Ẽ)
    # TODO: flip values at imaginary axes instead
    λ₋ = filter(l -> real(l) < 0, λ)

    return λ₋
end

