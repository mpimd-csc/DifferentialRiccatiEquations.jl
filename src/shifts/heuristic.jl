# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

import KernelAbstractions as KA
using UnPack

"""
    Shifts.Heuristic(nshifts, k₊, k₋; alg_E=Backslash(), alg_A=Backslash())

Compute heuristic or sub-optimal shift parameters following Algorithm 5.1 of

> Penzl: A cyclic low rank Smith method for large sparse Lyapunov equations,
> SIAM J. Sci. Comput., 21 (1999), pp. 1401-1418. DOI: 10.1137/S1064827598347666

Arguments:

- `nshifts::Int`: number of shifts to compute
- `k₊::Int`: number of Arnoldi iterations w.r.t. E⁻¹A
- `k₋::Int`: number of Arnoldi iterations w.r.t. A⁻¹E
- `alg_E::BlockLinearSolver`: linear solver inside Arnoldi iteration w.r.t. E⁻¹A
- `alg_A::BlockLinearSolver`: linear solver inside Arnoldi iteration w.r.t. A⁻¹E
"""
struct Heuristic <: Strategy
    nshifts::Int
    k₊::Int
    k₋::Int
    alg_E::BlockLinearSolver
    alg_A::BlockLinearSolver

    Heuristic(nshifts, k₊, k₋; alg_E=Backslash(), alg_A=Backslash()) = new(nshifts, k₊, k₋, alg_E, alg_A)
end

function Base.show(io::IO, ::MIME"text/plain", s::Heuristic)
    print(io, typeof(s), "(", s.nshifts, ", ", s.k₊, ", ", s.k₋)
    s.alg_E isa Backslash || print(io, ", alg_E=", s.alg_E)
    s.alg_A isa Backslash || print(io, ", alg_A=", s.alg_A)
    print(io, ")")
end

function init(strategy::Heuristic, prob)
    @unpack nshifts, k₊, k₋, alg_E, alg_A = strategy
    @unpack E, A = prob

    # TODO: Make solver configurable.
    # TODO: Think about caching of properties of E.
    # The matrix E shouldn't change all that much between iterations of the same algorithm,
    # or between algorithms in general.

    # Create a dense vector of ones on the same compute backend as A and E.
    b0 = arnoldi_b0(E)

    solver = CommonSolve.init(BlockLinearProblem(E, similar(b0)), alg_E)
    R₊ = compute_ritz_values(b0, k₊, "E⁻¹A") do x
        mul!(rhs(solver), A, x)
        solve!(solver)
    end
    solver = CommonSolve.init(BlockLinearProblem(A, similar(b0)), alg_A)
    R₋ = compute_ritz_values(b0, k₋, "A⁻¹E") do x
        mul!(rhs(solver), E, x)
        solve!(solver)
    end
    # TODO: R₊ and R₋ may not be disjoint. Remove duplicates, or replace values that differ
    # by an eps with their average.
    R = vcat(R₊, inv.(R₋))

    heuristic(R, nshifts)
end

"""
    arnoldi_b0(E) -> b0

Create a dense vector `b0` to start the Arnoldi process with.
The resulting vector must support `mul!(similar(b0), E, b0)`,
i.e. the data should be located on the same compute device.
"""
function arnoldi_b0(E)
    backend = KA.get_backend(E)
    T = eltype(E)
    n = size(E, 2)
    KA.ones(backend, T, n)
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

function compute_ritz_values(A, b0::AbstractVector, k::Int, desc::String)
    n = length(b0)
    H = zeros(k + 1, k)
    V = similar(b0, n, k + 1)
    V[:, 1] .= (1.0 / norm(b0)) * b0

    # Arnoldi
    for j in 1:k
        w = A(V[:, j])

        # Repeated modified Gram-Schmidt (MGS)
        for _ = 1:2
            for i = 1:j
                g = view(V, :, i)' * w
                H[i, j] += g
                w -= V[:, i] * g
            end
        end

        H[j+1, j] = beta = norm(w)
        V[:, j+1] .= (1.0 / beta) * w
    end

    ritz = eigvals(@view H[1:k, 1:k])

    stabilize_ritz_values!(ritz, desc)
end
