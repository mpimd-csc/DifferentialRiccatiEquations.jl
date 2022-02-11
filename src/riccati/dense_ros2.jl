function _solve(
    prob::GDREProblem{<:Matrix},
    alg::Ros2;
    dt::Real,
    save_state::Bool,
)
    @unpack E, A, B, C, tspan = prob
    X = prob.X0
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Global parameter for the method
    γ = 1+(1/sqrt(2))

    # Output Trajectories
    Xs = [X]
    save_state && sizehint!(Xs, len)
    K = (B'*X)*E
    Ks = [K]
    sizehint!(Ks, len)

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        gF = γ*τ*(A-B*K) - E/2
        Fs, Es, Q, Z = schur(gF, E)

        # Solve Lyapunov equation of 1st stage
        R = C'*C + A'*X*E + E'*X*A - K'*K
        R = real(R+R')/2
        utqu!(R, Z) # R = Z'*R*Z
        lyapcs!(Fs, Es, R; adj=true)
        K1 = R
        utqu!(K1, Q') # K1 = Q*K1*Q'

        # Solve Lyapunov equation of 2nd stage
        R2 = -τ^2*(E'*(K1*B))*((B'*K1)*E) - (2-1/γ)*E'*K1*E
        R2 = real(R2+R2')/2
        utqu!(R2, Z) # R2 = Z'*R2*Z
        lyapcs!(Fs, Es, R2; adj=true)
        K̃2 = R2
        utqu!(K̃2, Q') # K̃2 = Q*K̃2*Q'
        K2 = K̃2 + (4-1/γ)*K1

        # Update X
        X = X + (τ/2)*K2
        save_state && push!(Xs, X)

        # Update K
        K = (B'*X)*E
        push!(Ks, K)
    end
    save_state || push!(Xs, X)

    return DRESolution(Xs, Ks, tstops)
end
