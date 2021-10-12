function solve(
    prob::GDREProblem,
    alg::Ros3;
    dt::Real,
)
    @unpack E, A, B, C, tspan = prob
    X = prob.X0
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Output Trajectories
    Xs = [X]
    sizehint!(Xs, len)
    K = (B'*X)*E
    Ks = [K]
    sizehint!(Ks, len)

    # Global parameter for the method
    γ = 7.886751345948129e-1
    a21 = 1.267949192431123
    c21 = -1.607695154586736
    c31 = -3.464101615137755
    c32 = -1.732050807568877
    m1 = 2
    m2 = 5.773502691896258e-1
    m3 = 4.226497308103742e-1

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        gF = (A - B*K) - E/(2γ*τ)

        # Solve Lyapunov equation of 1st stage
        AXE = A'X*E
        R = C'C + AXE + AXE' - K'K
        R = real(R+R')/2
        K1 = lyapc(gF', E', R)

        # Solve Lyapunov equation of 2nd stage
        RX = (A'K1 - K'*(B'K1))*E
        R23 = a21*(RX+RX')
        R2 = R23 + (c21/τ)*E'K1*E
        R2 = real(R2+R2')/2
        K21 = lyapc(gF', E', R2)

        # Solve Lyapunov equation of 3rd stage
        R3 = R23 + E'*(((c31/τ)+(c32/τ))*K1 + (c32/τ)*K21)*E
        R3 = real(R3+R3')/2
        K31 = lyapc(gF', E', R3)

        # Update X
        X = X + (m1+m2+m3)*K1 + m2*K21 + m3*K31
        push!(Xs, X)

        # Update K
        K = (B'*X)*E
        push!(Ks, K)
    end

    return DRESolution(Xs, Ks, tstops)
end
