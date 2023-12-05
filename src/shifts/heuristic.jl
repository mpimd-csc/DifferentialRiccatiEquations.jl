# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using UnPack
using ArnoldiMethod: partialschur
using LinearMaps: InverseMap, LinearMap

"""
    Shifts.Heuristic(nshifts, k₊, k₋)

Compute heuristic or sub-optimal shift parameters following Algorithm 5.1 of

> Penzl: A cyclic low rank Smith method for large sparse Lyapunov equations,
> SIAM J. Sci. Comput., 21 (2000), pp. 1401-1418. DOI: 10.1137/S1064827598347666
"""
struct Heuristic <: Strategy
    nshifts::Int
    k₊::Int
    k₋::Int
end

function init(strategy::Heuristic, prob)
    @unpack nshifts, k₊, k₋ = strategy
    @unpack E, A = prob

    # TODO: Make solver configurable.
    # TODO: Think about caching of properties of E.
    # The matrix E shouldn't change all that much between iterations of the same algorithm,
    # or between algorithms in general.
    solver(y, A, x) = (y .= A \ x)
    E⁻¹A = InverseMap(E; solver) * LinearMap{eltype(A)}(x -> A*x, size(A)...)
    A⁻¹E = InverseMap(A; solver) * LinearMap(E)
    R₊ = compute_ritz_values(E⁻¹A, k₊)
    R₋ = compute_ritz_values(A⁻¹E, k₋)
    # TODO: R₊ and R₋ may not be disjoint. Remove duplicates, or replace values that differ
    # by an eps with their average.
    R = vcat(R₊, inv.(R₋))

    s(t, P) = prod(abs(t - p) / abs(t + p) for p in P)

    p = argmin(R) do p
        maximum(s(t, (p,)) for t in R)
    end
    P = isreal(p) ? [p] : [p, conj(p)]
    while length(P) < nshifts
        p = argmax(R) do t
            s(t, P)
        end
        if isreal(p)
            push!(P, p)
        else
            append!(P, (p, conj(p)))
        end
    end

    return P
end

function compute_ritz_values(A, n::Int)
    decomp, _ = partialschur(A; nev=n)
    decomp.eigenvalues
end
