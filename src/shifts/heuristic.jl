# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using UnPack

"""
    Shifts.Heuristic(nshifts, k₊, k₋)

Compute heuristic or sub-optimal shift parameters following Algorithm 5.1 of

> Penzl: A cyclic low rank Smith method for large sparse Lyapunov equations,
> SIAM J. Sci. Comput., 21 (1999), pp. 1401-1418. DOI: 10.1137/S1064827598347666
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
    Ef = factorize(E)

    b0 = ones(size(E, 1))
    R₊ = compute_ritz_values(x -> Ef \ (A * x), b0, k₊, "E⁻¹A")
    R₋ = compute_ritz_values(x -> A \ (E * x), b0, k₋, "A⁻¹E")
    # TODO: R₊ and R₋ may not be disjoint. Remove duplicates, or replace values that differ
    # by an eps with their average.
    R = vcat(R₊, inv.(R₋))

    heuristic(R, nshifts)
end

function heuristic(R, nshifts=length(R))
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

function compute_ritz_values(A, b0, k::Int, desc::String)
    n = length(b0)
    H = zeros(k + 1, k)
    V = zeros(n, k + 1)
    V[:, 1] .= (1.0 / norm(b0)) * b0

    # Arnoldi
    for j in 1:k
        w = A(V[:, j])

        # Repeated modified Gram-Schmidt (MGS)
        for _ = 1:2
            for i = 1:j
                g = V[:, i]' * w
                H[i, j] += g
                w -= V[:, i] * g
            end
        end

        H[j+1, j] = beta = norm(w)
        V[:, j+1] .= (1.0 / beta) * w
    end

    ritz = eigvals(@view H[1:k, 1:k])

    isstable(v) = real(v) < 0
    all(isstable, ritz) && return ritz

    @warn "Discarding unstable Ritz values of $desc"
    filter!(isstable, ritz)
end
