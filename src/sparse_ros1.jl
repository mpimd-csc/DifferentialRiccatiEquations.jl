function _solve(
    prob::GDREProblem{T},
    ::Ros1;
    dt::Real,
    save_state::Bool,
) where {T <: LDLᵀ}
    @unpack E, A, B, C, tspan = prob
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

    TL = typeof(X.L)
    TD = typeof(X.D)

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        # Coefficient Matrix of the Lyapunov Equation
        F = lr_update(A - E/(2τ), -1, B, K)

        # Right-hand side:
        G::TL = [C' E'L]
        n_G = size(G, 2)
        S::TD = _zeros(TD, n_G)
        q = size(C, 1)
        S[1:q, 1:q] = I(q)
        S[q+1:end, q+1:end] = (BᵀLD)' * BᵀLD + D/τ
        R::T = compress(LDLᵀ(G, S))

        # Update X
        lyap = GALEProblem(E, F, R)
        X = solve(lyap, ADI())
        save_state && push!(Xs, X)

        # Update K
        L, D = X
        BᵀLD = (B'*L)*D
        K = BᵀLD*(L'*E)
        push!(Ks, K)
    end
    save_state || push!(Xs, X)

    return DRESolution(Xs, Ks, tstops)
end
