# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _solve(
    prob::GDREProblem{LDLᵀ{TL,TD}},
    alg::Ros2;
    dt::Real,
    save_state::Bool,
    observer,
) where {TL,TD}
    observe_gdre_start!(observer, prob, alg)

    @unpack E, A, B, C, tspan = prob
    q = size(C, 1)
    X = prob.X0
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Global parameter for the method
    γ = 1+(1/sqrt(2))

    # Output Trajectories
    Xs = [X]
    save_state && sizehint!(Xs, len)
    L, D = X
    BᵀLD = (B'*L)*D
    K = BᵀLD*(L'*E)
    Ks = [K]
    sizehint!(Ks, len)

    observe_gdre_step!(observer, tstops[1], X, K)

    inner_alg = @something(alg.inner_alg, ADI())
    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        γτ = γ*τ
        F = lr_update(γτ*A - E/2, inv(-γτ), B, K)

        # Solve Lyapunov equation of 1st stage
        G::TL = _hcat(TL, C', A'L, E'L)
        n_G = size(G, 2)
        n_L = size(L, 2)
        S::TD = _zeros(TD, n_G)
        b1 = 1:q
        b2 = q+1:q+n_L
        b3 = n_G-n_L+1:n_G
        S[b1, b1] = I(q)
        S[b2, b3] = D
        S[b3, b2] = D
        S[b3, b3] = - (BᵀLD)' * BᵀLD
        R1 = compress!(LDLᵀ{TL,TD}(G, S))

        lyap = GALEProblem(E, F, R1)
        K1 = solve(lyap, inner_alg; observer)

        # Solve Lyapunov equation of 2nd stage
        T₁, D₁ = K1
        BᵀT₁D₁ = (B'*T₁)*D₁
        G₂::TL = E'T₁
        S₂::TD = (τ^2 * BᵀT₁D₁)' * BᵀT₁D₁ + (2-1/γ) * D₁
        R2 = LDLᵀ{TL,TD}(G₂, S₂)

        lyap = GALEProblem(E, F, R2)
        K2 = solve(lyap, inner_alg; observer)

        # Update X
        X = X + ((2-1/2γ)*τ)*K1 + (-τ/2)*K2
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
