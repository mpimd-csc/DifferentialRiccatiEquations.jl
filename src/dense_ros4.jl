function solve(
    prob::GDREProblem,
    alg::Ros4;
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

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        gF = (τ*(A-B*K)-E)/2

        # Solve Lyapunov equation of 1st stage
        AXE = A'X*E
        R = C'C + AXE + AXE' - K'K
        R = real(R+R')/2
        K1 = lyapc(gF', E', R)

        # Solve Lyapunov equation of 2nd stage
        EK1E = E'*K1*E
        EK1B = E'*(K1*B)
        R2 = -τ^2*(EK1B*EK1B')-2*EK1E
        R2 = real(R2+R2')/2
        K21 = lyapc(gF', E', R2)
        K2 = K21 - K1

        # Solve Lyapunov equation of 3rd stage
        α = (24/25)*τ
        β = (3/25)*τ
        EK2E = E'*K2*E
        EK2B = E'*(K2*B)
        TMP = EK2B*EK1B'
        R3 = (245/25)*EK1E + (36/25)*EK2E - (426/625)*τ^2*(EK1B*EK1B') - β^2*(EK2B*EK2B') - α*β*(TMP+TMP')
        R3 = real(R3+R3')/2
        K31 = lyapc(gF', E', R3)
        K3 = K31 - (17/25)*K1

        # Solve Lyapunov equation of 4th stage
        R4 = -(981/125)*EK1E-(177/125)*EK2E-(1/5)*E'*K3*E
        R4 = real(R4+R4')/2
        K41 = lyapc(gF', E', R4)
        K4 = K41 + K3

        # Update X
        X = X + τ*((19/18)*K1 + 0.25*K2 + (25/216)*K3 + (125/216)*K4)
        push!(Xs, X)

        # Update K
        K = (B'*X)*E
        push!(Ks, K)
    end

    return DRESolution(Xs, Ks, tstops)
end
