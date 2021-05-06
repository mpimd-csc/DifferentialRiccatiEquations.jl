function solve(
    prob::GDREProblem,
    alg::Ros1;
    dt::Real,
)
    @unpack E, A, B, C, tspan = prob
    X = prob.X0
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Output Trajectory
    Xs = [X]
    sizehint!(Xs, len)

    for i in 2:len
        τ = tstops[i-1] - tstops[i]
        K = (B'*X)*E

        # Coefficient Matrix of the Lyapunov Equation
        F = (A-B*K) - E/(2τ)
        R = C'*C + K'*K + (1/τ)*E'*X*E

        # Only for safety
        R = real(R+R')/2

        # Solve the Equation
        X = lyapc(F', E', R)

        # Store X
        push!(Xs, X)
    end

    return DRESolution(Xs, tstops)
end
