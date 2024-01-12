# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(
    prob::GAREProblem{TG,TQ},
    ::NewtonADI;
    reltol = size(prob.A, 1) * eps(),
    maxiters = 5,
    observer = nothing,
    adi_initprev::Bool = false,
    adi_kwargs::NamedTuple = NamedTuple(),
) where {TG,TQ}
    TG <: LDLᵀ{<:AbstractMatrix,UniformScaling{Bool}} || error("TG=$TG not yet implemented")
    TQ <: LDLᵀ{<:AbstractMatrix,UniformScaling{Bool}} || error("TQ=$TQ not yet implemented")

    observe_gare_start!(observer, prob, NewtonADI())
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
        observe_gare_step!(observer, i, X, res, res_norm)

        res_norm <= abstol && break
        if i > maxiters
            observe_gare_failed!(observer)
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
        initial_guess = adi_initprev ? X : nothing

        # Newton step:
        lyap = GALEProblem(E, F, RHS)
        X = solve(lyap, ADI(); maxiters=100, reltol, observer, initial_guess, adi_kwargs...)
    end

    observe_gare_done!(observer, i, X, res, res_norm)
    X
end
