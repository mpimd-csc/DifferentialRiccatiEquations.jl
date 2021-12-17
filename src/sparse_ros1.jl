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

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        # Coefficient Matrix of the Lyapunov Equation
        F = (A-B*K) - E/(2τ)

        # Right-hand side:
        G = [C' L]
        l = size(BᵀLD, 2)
        q = size(G, 2) - l
        S = similar(G, q+l, q+l)
        S[1:q, 1:q] .= I(q)
        S[q+1:end, q+1:end] = (BᵀLD)' * BᵀLD + D/τ
        R = LDLᵀ(G, S)

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
