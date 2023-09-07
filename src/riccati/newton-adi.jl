# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(
    prob::GAREProblem{TG,TQ},
    ::NewtonADI;
    reltol = size(prob.A, 1) * eps(),
    maxiters = 5,
    observer = nothing,
    variant::Symbol = :zero,
) where {TG,TQ}
    TG <: LDLᵀ{<:AbstractMatrix,UniformScaling{Bool}} || error("TG=$TG not yet implemented")
    TQ <: LDLᵀ{<:AbstractMatrix,UniformScaling{Bool}} || error("TQ=$TQ not yet implemented")
    TL = TD = Matrix{Float64}

    @unpack E, A, Q = prob
    B, _ = prob.G
    Cᵀ, _ = Q

    n = size(A, 2)
    X = LDLᵀ{TL,TD}(zeros(n, 0), zeros(0, 0)) # this is ugly

    abstol = reltol * norm(Q)
    i = 0
    while true
        # Compute residual
        L, D = X
        AᵀL = A'L
        EᵀL = E'L
        BᵀLD = (B'L)*D
        DLᵀGLD = (BᵀLD)'BᵀLD
        K = BᵀLD * (EᵀL)'

        res = norm(residual(prob, X; AᵀL, EᵀL, DLᵀGLD))
        @debug "NewtonADI" i rank(X) residual=res abstol reltol
        res <= abstol && break
        if i > maxiters
            @warn "NewtonADI did not converge"
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
        if variant == :zero
            initial_guess = nothing
        elseif variant == :prev
            initial_guess = X
        else
            error("unkown variant $variant")
        end

        # Newton step:
        lyap = GALEProblem(E, F, RHS)
        X = solve(lyap, ADI(); maxiters=100, reltol, observer, initial_guess)
    end

    @debug "NewtonADI done" steps=i
    X
end
