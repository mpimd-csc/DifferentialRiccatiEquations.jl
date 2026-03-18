# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

# TODO: Refactor below BiCGStab to allow `size(b, 2) > 1`:
# - Remove variable `h`
# - Replace `alpha * p` by `p * Diagonal(alpha)` etc.
# - Use `SimpleBiCGStab()` to solve `BlockLinearProblem`

# TODO: Should every `BlockLinearSolver` be able to solve `LinearProblem`s?
# While convenient during testing, it may be too much to ask from our users.

using DifferentialRiccatiEquations: LinearProblem
using Random: AbstractRNG, default_rng

import CommonSolve
import DifferentialRiccatiEquations: rhs, solution

@kwdef struct SimpleBiCGStab
    maxiters::Int = 100
    reltol::Float64 = 100eps()
    rng::AbstractRNG = default_rng()
end

struct SimpleBiCGStabCache{T}
    prob::LinearProblem
    alg::SimpleBiCGStab
    x::T
    h::T
    s::T
    r::T
    v::T
    p::T
    t::T
    r0::T
    # Used by inexact ADI:
    abstol
    reltol
end

rhs(cache::SimpleBiCGStabCache) = rhs(cache.prob)
solution(cache::SimpleBiCGStabCache) = cache.x

function CommonSolve.init(prob::LinearProblem, alg::SimpleBiCGStab; abstol=nothing, reltol=nothing)
    (; A, b) = prob

    # For better reproducibility and predictability:
    rng = copy(alg.rng)

    # Pre-allocate all vectors:
    x = similar(b, eltype(A))
    h = similar(x)
    s = similar(x)
    r = similar(x)
    v = similar(x)
    p = similar(x)
    t = similar(x)

    # Random r0 can improve stability:
    r0 = randn(rng, eltype(A), length(b))
    r0 ./= norm(r0)

    SimpleBiCGStabCache{typeof(x)}(prob, alg, x, h, s, r, v, p, t, r0, abstol, reltol)
end

# Derived from https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
function CommonSolve.solve!(cache::SimpleBiCGStabCache)
    (; A, b) = cache.prob
    (; maxiters) = cache.alg
    (; x, h, s, r, r0, v, p, t) = cache

    # Use zero as initial guess:
    fill!(x, zero(eltype(x)))
    copyto!(r, b)
    copyto!(p, r)

    # Assume that A is regular:
    # This short-cut is needed for the Newton-ADI .... and I am not sure why.
    iszero(r) && return x

    # Initialize tolerances:
    reltol = @something(cache.reltol, cache.alg.reltol, size(A, 1) * eps(eltype(b)))
    abstol = @something(cache.abstol, reltol * norm(r))

    rho = dot(r0, r)
    for _ in 1:maxiters
        mul!(v, A, p)
        alpha = rho / dot(r0, v)
        @. h = x + alpha * p
        @. s = r - alpha * v
        if norm(s) <= abstol
            copyto!(x, h)
            break
        end
        mul!(t, A, s)
        omega = dot(t, s) / dot(t, t)
        @. x = h + omega * s
        @. r = s - omega * t
        if norm(r) <= abstol
            break
        end
        rho_next = dot(r0, r)
        beta = (rho_next / rho) * (alpha / omega)
        @. p = r + beta * (p - omega * v)
        rho = rho_next
    end

    return x
end
