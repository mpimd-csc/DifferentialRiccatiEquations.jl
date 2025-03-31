# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(
    prob::GAREProblem{TG,TQ},
    alg::Newton;
    observer = nothing,
) where {TG,TQ}
    eltype(prob.G.Ds) == UniformScaling{Bool} || error("TG=$TG not yet implemented")
    eltype(prob.Q.Ds) == UniformScaling{Bool} || error("TQ=$TQ not yet implemented")

    @timeit_debug "callbacks" observe_gare_start!(observer, prob, alg)
    TL = TD = Matrix{Float64}

    @unpack E, A, Q = prob
    alpha, B, _ = prob.G
    alpha == 1 || error("Scaled prob.G not yet implemented")
    alpha, Cᵀ, _ = Q
    alpha == 1 || error("Scaled prob.Q not yet implemented")
    res = Q
    res_norm = norm(res)
    reltol = @something(alg.reltol, size(A, 1) * eps())
    abstol = @something(alg.abstol, reltol * res_norm)

    # This is ugly:
    n = size(A, 2)
    L = convert(TL, zeros(n, 0))::TL
    D = convert(TD, zeros(0, 0))::TD
    X = lowrank(L, D)

    i = 0
    local X_prev
    @unpack maxiters, inner_alg = alg
    inner_reltol = unpack(inner_alg, Val(:reltol))
    inner_reltol = @something(inner_reltol, reltol / 10)
    @unpack inexact, inexact_hybrid, inexact_forcing, linesearch = alg
    while true
        # Compute residual
        alpha, L, D = X
        EᵀL = E'L
        BᵀLD = (B'L)*D
        alpha == 1 || rmul!(BᵀLD, alpha)
        DLᵀGLD = (BᵀLD)'BᵀLD
        K = BᵀLD * (EᵀL)'

        res = residual(prob, X; EᵀL, DLᵀGLD)
        res_norm_prev = res_norm
        res_norm = norm(res)

        if i > 0 && linesearch
            @timeit_debug "Armijo line search" begin
                α = 0.1
                # The line search is mostly triggered for early Newton iterations `i`,
                # where the linear systems to be solved have few columns and `X` has low rank.
                # Therefore, an efficient implementation is not that important for now.
                if res_norm > (1-α) * res_norm_prev
                    X̃ = X # backup if line search fails
                    β = 1/2 # Armijo parameter
                    λ = β # step size
                    while true
                        # Check sufficient decrease condition:
                        # (naive implementation)
                        X = (1 - λ) * X_prev + λ * X̃
                        res = residual(prob, X)
                        res_norm = norm(res)
                        if res_norm < (1 - λ*α) * res_norm_prev
                            @debug "Accepting line search λ=$λ"
                            # Update feedback matrix K and other auxillary variables:
                            # (naive implementation)
                            alpha, L, D = X
                            EᵀL = E'L
                            BᵀLD = (B'L)*D
                            alpha == 1 || rmul!(BᵀLD, alpha)
                            DLᵀGLD = (BᵀLD)'BᵀLD
                            K .= BᵀLD * (EᵀL)'
                            break
                        end

                        # Prepare next step size:
                        λ *= β
                        if λ < eps()
                            @warn "Line search failed; using un-modified iterate"
                            λ = 1.0
                            X = X̃
                            break
                        end
                    end
                    @timeit_debug "callbacks" observe_gare_metadata!(observer, "line search", λ)
                end
            end
        end
        @timeit_debug "callbacks" observe_gare_step!(observer, i, X, res, res_norm)
        @debug "Newton" i reltol abstol residual=res_norm rank(X)

        res_norm <= abstol && break
        if i >= maxiters
            @timeit_debug "callbacks" observe_gare_failed!(observer)
            @warn "Newton method did not converge" residual=res_norm abstol maxiters
            break
        end
        i += 1

        # Coefficient Matrix of the Lyapunov Equation
        F = lr_update(A, -1, B, K)

        # Right-hand side:
        m = size(B, 2)
        q = size(Cᵀ, 2)
        EᵀXB = EᵀL * (BᵀLD)'
        G::TL = _hcat(TL, Cᵀ, EᵀXB)
        S::TD = _dcat(TD, (I(q), I(m)))
        RHS = lowrank(G, S)

        # Lyapunov setup
        lyap = GALEProblem(E, F, RHS)
        if inexact
            η = inexact_forcing(i, res_norm)
            inner_abstol = η * res_norm
            if inexact_hybrid
                # If the classical/"exact" tolerance is less strict than
                # the one of the Inexact Newton, use that tolerance instead.
                classical_abstol = inner_reltol * norm(lyap.C)
                switch_back = classical_abstol > inner_abstol
                @timeit_debug "callbacks" observe_gare_metadata!(observer, "inexact", !switch_back)
                if switch_back
                    @debug "Switching from inexact to classical Newton method" i inexact_abstol=inner_abstol classical_abstol
                    inner_abstol = classical_abstol
                end
            else
                @timeit_debug "callbacks" observe_gare_metadata!(observer, "inexact", true)
            end
        else
            inner_abstol = inner_reltol * norm(lyap.C)
        end

        # Newton step:
        X_prev = X
        X = @timeit_debug "$(typeof(inner_alg))" solve(
            lyap, inner_alg;
            abstol=inner_abstol,
            initial_guess=X_prev,
            observer,
        )
    end

    @timeit_debug "callbacks" observe_gare_done!(observer, i, X, res, res_norm)
    X
end

"""
    superlinear_forcing(i, _) = 1 / (i^3 + 1)

Exemplary forcing term to obtain superlinear convergence in the inexact Newton method.
`i::Int` refers to the current Newton step.
See [`Newton`](@ref).
"""
superlinear_forcing(i, _) = 1 / (i^3 + 1)

"""
    quadratic_forcing(_, residual_norm) = min(0.1, 0.9 * residual_norm)

Exemplary forcing term to obtain quadratic convergence in the inexact Newton method.
`residual_norm::Float64` refers to the norm of the previous Newton residual.
See [`Newton`](@ref).
"""
quadratic_forcing(_, residual_norm) = min(0.1, 0.9 * residual_norm)
