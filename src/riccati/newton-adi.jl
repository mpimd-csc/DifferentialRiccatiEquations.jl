# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

# Keep defaults in sync with docstring of NewtonADI!
function CommonSolve.solve(
    prob::GAREProblem{TG,TQ},
    ::NewtonADI;
    reltol = size(prob.A, 1) * eps(),
    maxiters = 5,
    observer = nothing,
    adi_initprev::Bool = false,
    adi_kwargs::NamedTuple = NamedTuple(),
    inexact::Bool = true,
    inexact_hybrid::Bool = true,
    inexact_forcing = quadratic_forcing,
) where {TG,TQ}
    TG <: LDLᵀ{<:AbstractMatrix,UniformScaling{Bool}} || error("TG=$TG not yet implemented")
    TQ <: LDLᵀ{<:AbstractMatrix,UniformScaling{Bool}} || error("TQ=$TQ not yet implemented")

    @timeit_debug "callbacks" observe_gare_start!(observer, prob, NewtonADI())
    TL = TD = Matrix{Float64}

    @unpack E, A, Q = prob
    B, _ = prob.G
    Cᵀ, _ = Q

    n = size(A, 2)
    X = LDLᵀ{TL,TD}(zeros(n, 0), zeros(0, 0)) # this is ugly

    res = Q
    res_norm = norm(res)
    abstol = reltol * res_norm

    i = 0
    while true
        # Compute residual
        L, D = X
        AᵀL = A'L
        EᵀL = E'L
        BᵀLD = (B'L)*D
        DLᵀGLD = (BᵀLD)'BᵀLD
        K = BᵀLD * (EᵀL)'

        res = residual(prob, X; AᵀL, EᵀL, DLᵀGLD)
        res_norm = norm(res)
        @timeit_debug "callbacks" observe_gare_step!(observer, i, X, res, res_norm)

        res_norm <= abstol && break
        if i >= maxiters
            @timeit_debug "callbacks" observe_gare_failed!(observer)
            @warn "NewtonADI did not converge" residual=res_norm abstol maxiters
            break
        end
        i += 1

        # Coefficient Matrix of the Lyapunov Equation
        F = lr_update(A, -1, B, K)

        # Right-hand side:
        q = size(Cᵀ, 2)
        G::TL = _hcat(TL, Cᵀ, EᵀL)
        S::TD = _dcat(TD, I(q), DLᵀGLD)
        RHS = LDLᵀ(G, S)
        compress!(RHS)

        # ADI setup
        lyap = GALEProblem(E, F, RHS)
        initial_guess = adi_initprev ? X : nothing
        adi_reltol = get(adi_kwargs, :reltol, reltol / 10)
        if inexact
            η = inexact_forcing(i, res_norm)
            adi_abstol = η * res_norm
            if inexact_hybrid
                # If the classical/"exact" tolerance is less strict than
                # the one of the Inexact Newton, use that tolerance instead.
                classical_abstol = adi_reltol * norm(lyap.C)
                if classical_abstol > adi_abstol
                    @debug "Switching from inexact to classical Newton method" i inexact_abstol=adi_abstol classical_abstol
                    adi_abstol = classical_abstol
                end
            end
        else
            adi_abstol = adi_reltol * norm(lyap.C)
        end

        # Newton step:
        X = @timeit_debug "ADI" solve(
            lyap, ADI();
            maxiters=100,
            observer,
            initial_guess,
            abstol=adi_abstol,
            adi_kwargs...)
    end

    @timeit_debug "callbacks" observe_gare_done!(observer, i, X, res, res_norm)
    X
end

"""
    superlinear_forcing(i, _) = 1 / (i^3 + 1)

Exemplary forcing term to obtain superlinear convergence in the inexact Newton method.
`i::Int` refers to the current Newton step.
See [`NewtonADI`](@ref).
"""
superlinear_forcing(i, _) = 1 / (i^3 + 1)

"""
    quadratic_forcing(_, residual_norm) = min(0.1, 0.9 * residual_norm)

Exemplary forcing term to obtain quadratic convergence in the inexact Newton method.
`residual_norm::Float64` refers to the norm of the previous Newton residual.
See [`NewtonADI`](@ref).
"""
quadratic_forcing(_, residual_norm) = min(0.1, 0.9 * residual_norm)
