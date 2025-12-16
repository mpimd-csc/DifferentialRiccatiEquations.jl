# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

# - Implementation follows Algorithm 2.2 of original FGMRES paper: https://doi.org/10.1137/0914028
# - Linear operator is called `L` instead of `A`
# - Not yet implemented: `P_l` and `P_r` described in Algorithm 5.4 of https://doi.org/10.1007/978-3-319-07236-4_5

function CommonSolve.solve(
    prob::GALEProblem,
    alg::GMRES;
    initial_guess::Union{Nothing,<:LDLᵀ}=nothing,
    abstol=nothing,
    observer=nothing,
)
    @timeit_debug "callbacks" observe_gale_start!(observer, prob, alg)
    (; E, A, C) = prob
    (; maxiters, maxrestarts, compression) = alg
    L = LyapunovOperator(E, A)
    if alg.ignore_initial_guess || initial_guess === nothing
        initial_guess = zero(C)
    end
    X = initial_guess

    # Initialize tolerances
    reltol = @something(alg.reltol, size(A, 1) * eps(eltype(X)))
    abstol = @something(abstol, alg.abstol, reltol * norm(C)) # use same tolerance as if initial_guess=zero(C)

    preconditioner = specialize(alg.preconditioner, prob)
    H = zeros(maxiters + 1, maxiters)
    b = zeros(maxiters + 1)
    V = Vector(undef, maxiters + 1)
    Z = Vector(undef, maxiters)

    local m, residual_norm, restarts
    for outer restarts in 0:maxrestarts
        m = 0
        R0 = residual(prob, X)
        beta = residual_norm = norm(R0)
        @timeit_debug "callbacks" observe_gale_step!(observer, 0, X, R0, beta)
        beta <= abstol && break

        # Arnoldi process:
        V .= nothing
        Z .= nothing
        V[1] = R0 / beta
        b[1] = beta
        local y
        for j in 1:maxiters
            if preconditioner === nothing
                Z[j] = V[j]
            else
                Z[j] = solve(GALEProblem(E, A, V[j]), preconditioner; observer)
            end
            W = L * Z[j]
            compression && compress!(W)
            for i in 1:j
                H[i, j] = dot(V[i], W)
                W -= H[i, j] * V[i]
            end
            H[j + 1, j] = norm(W)
            V[j + 1] = W / H[j + 1, j]
            # Defer compression of V[j + 1] until start of next loop.

            # Find `y` that minimizes `|b - Hy|`:
            m = j
            Hₘ = @view H[1:m+1, 1:m]
            bₘ = @view b[1:m+1]
            y = Hₘ \ bₘ

            # Compute residual of corresponding solution:
            # Avoid assembly of solution X; it is only needed if the algorithm has converged.
            # The following quantity may differ slightly (e.g., due to low-rank compression),
            # but usually it doesn't differ much.
            residual_norm = norm(bₘ - Hₘ * y)
            @debug "GMRES $j" rank(V[j]) rank(Z[j]) residual_gap=residual_norm/abstol
            residual_norm <= abstol && break

            # Avoid issuing the callback twice for iteration `m`, i.e., keep this after `break`:
            @timeit_debug "callbacks" observe_gale_step!(observer, m, nothing, nothing, residual_norm)

            # The next basis vector V[j + 1] will be needed in the next GMRES iteration:
            compression && compress!(V[j + 1])
        end

        # Form the approximate solution:
        # Recall that our `residual()` computes `Ax - b` instead of `b - Ax`,
        # negating all entries of `V` and `Z`, thus requiring `X - sum(...)`
        # instead of `X + sum(...)`. Use formulation robust to `m == 0`.
        X = sum(-y[j] * Z[j] for j in 1:m; init=X)
        compression && compress!(X)
        @timeit_debug "callbacks" observe_gale_step!(observer, m, X, nothing, residual_norm)

        residual_norm <= abstol && break
    end

    if residual_norm > abstol
        @timeit_debug "callbacks" observe_gale_failed!(observer)
        @warn "GMRES did not converge" residual=residual_norm abstol maxrestarts maxiters
    end

    # Compute the total number of iterations completed:
    iters = restarts * maxiters + m
    @timeit_debug "callbacks" observe_gale_done!(observer, iters, X, nothing, residual_norm)
    @debug "GMRES done" residual=residual_norm abstol m iters restarts

    return X
end

struct LyapunovOperator{TE, TA}
    E::TE
    A::TA
end

function Base.:(*)(L::LyapunovOperator, X::LDLᵀ)
    (; E, A) = L
    a, Z, Y = X
    O = zero(Y)
    Z2 = [E'Z A'Z]
    Y2 = [O Y; Y O]
    a * lowrank(Z2, Y2)
end

specialize(x, _) = x
specialize(c::Shifts.Cyclic, prob) = Shifts.Cyclic(specialize(c.inner, prob))
specialize(h::Shifts.Heuristic, prob) = Shifts.init(h, prob)

function specialize(adi::ADI, prob)
    shifts = specialize(adi.shifts, prob)
    setproperties(adi; shifts)
end

function specialize(gmres::GMRES, prob)
    preconditioner = specialize(gmres.preconditioner, prob)
    setproperties(gmres; preconditioner)
end
