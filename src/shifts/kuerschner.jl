# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using LinearAlgebra, SparseArrays
using UnPack

"""
    KuerschnerV(prob, u::Int)

Compute shift parameters based on the `u` most recent increments comprising the solution candidate to `prob`.

It is recommended to use even `u > 1`, such that an ADI double-step can properly be accounted for.
"""
mutable struct KuerschnerV <: ShiftIterator
    prob
    n_history::Int
    Vs::Vector{Any}
    shifts::Vector{ComplexF64}

    function KuerschnerV(prob, u::Int)
        new(prob, u, [], ComplexF64[])
    end
end

function update_shifts!(it::KuerschnerV, _, R, Vs...)
    isempty(Vs) && push!(it.Vs, R)
    append!(it.Vs, Vs)
    lst = length(it.Vs)
    fst = max(1, lst - it.n_history + 1)
    keepat!(it.Vs, fst:lst)
    return
end

function compute_next_shifts!(it::KuerschnerV)
    @unpack E, A = it.prob
    @unpack Vs = it

    N = hcat(Vs...)::AbstractMatrix{<:Real}
    Q = orth(N)
    Ẽ = restrict(E, Q)
    Ã = restrict(A, Q)
    λ = eigvals(Ã, Ẽ)
    # TODO: flip values at imaginary axes instead
    λ₋ = filter(l -> real(l) < 0, λ)

    # Individual shifts will be extracted using pop!(),
    # so reverse them to not change the current behavior.
    reverse!(λ₋)

    return λ₋
end

