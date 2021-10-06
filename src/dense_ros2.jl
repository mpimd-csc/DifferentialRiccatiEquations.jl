function solve(
    prob::GDREProblem,
    alg::Ros2;
    dt::Real,
)
    @unpack E, A, B, C, tspan = prob
    X = prob.X0
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Global parameter for the method
    γ = 1+(1/sqrt(2))

    # Output Trajectories
    Xs = [X]
    sizehint!(Xs, len)
    K = (B'*X)*E
    Ks = [K]
    sizehint!(Ks, len)

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        gF = γ*τ*(A-B*K) - E/2

        # Solve Lyapunov equation of 1st stage
        R = C'*C + A'*X*E + E'*X*A - K'*K
        R = real(R+R')/2

        K1 = lyapc(gF', E', R)

        # Solve Lyapunov equation of 2nd stage
        R2 = -τ^2*(E'*(K1*B))*((B'*K1)*E) - (2-1/γ)*E'*K1*E
        R2 = real(R2+R2')/2

        K̃2 = lyapc(gF', E', R2)
        K2 = K̃2 + (4-1/γ)*K1

        # Update X
        X = X + (τ/2)*K2
        push!(Xs, X)

        # Update K
        K = (B'*X)*E
        push!(Ks, K)
    end

    return DRESolution(Xs, Ks, tstops)
end
