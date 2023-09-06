# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _solve(
    prob::GDREProblem{LDLᵀ{TL,TD}},
    ::Ros1;
    dt::Real,
    save_state::Bool,
    observer,
) where {TL,TD}
    T = LDLᵀ{TL,TD}

    @unpack E, A, B, C, tspan = prob
    q = size(C, 1)
    X = prob.X0::T
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Output Trajectories
    Xs = [X]
    save_state && sizehint!(Xs, len)
    L, D = X
    BᵀLD = (B'*L)*D
    K = BᵀLD*(L'*E)
    Ks = [K]
    sizehint!(Ks, len)

    observe_gdre_start!(observer, prob, Ros1())
    observe_gdre_step!(observer, tstops[1], X, K)

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        # Coefficient Matrix of the Lyapunov Equation
        F = lr_update(A - E/(2τ), -1, B, K)

        # Right-hand side:
        G::TL = _hcat(TL, C', E'L)
        S::TD = _dcat(TD, I(q), (BᵀLD)' * BᵀLD + D/τ)
        R::T = compress!(LDLᵀ(G, S))

        # Update X
        lyap = GALEProblem(E, F, R)
        X = solve(lyap, ADI(); observer, initial_guess=X)
        save_state && push!(Xs, X)

        # Update K
        L, D = X
        BᵀLD = (B'*L)*D
        K = BᵀLD*(L'*E)
        push!(Ks, K)

        observe_gdre_step!(observer, tstops[i], X, K)
    end
    save_state || push!(Xs, X)

    observe_gdre_done!(observer)

    return DRESolution(Xs, Ks, tstops)
end
