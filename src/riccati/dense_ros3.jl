function _solve(
    prob::GDREProblem{<:Matrix},
    alg::Ros3;
    dt::Real,
    save_state::Bool,
)
    @unpack E, A, B, C, tspan = prob
    Ed = collect(E)
    X = prob.X0
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Output Trajectories
    Xs = [X]
    save_state && sizehint!(Xs, len)
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

    CᵀC = C'C
    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        gF = (A - B*K) - E/(2γ*τ)
        Fs, Es, Q, Z = schur(gF, Ed)

        # Solve Lyapunov equation of 1st stage
        AXE = A'X*E
        R = CᵀC + AXE + AXE' - K'K
        R = real(R+R')/2
        utqu!(R, Z) # R = Z'*R*Z
        lyapcs!(Fs, Es, R; adj=true)
        K1 = R
        utqu!(K1, Q') # K1 = Q*K1*Q'

        # Solve Lyapunov equation of 2nd stage
        RX = (A'K1 - K'*(B'K1))*E
        R23 = a21*(RX+RX')
        R2 = R23 + (c21/τ)*E'K1*E
        R2 = real(R2+R2')/2
        utqu!(R2, Z) # R2 = Z'*R2*Z
        lyapcs!(Fs, Es, R2; adj=true)
        K21 = R2
        utqu!(K21, Q') # K21 = Q*K21*Q'

        # Solve Lyapunov equation of 3rd stage
        R3 = R23 + E'*(((c31/τ)+(c32/τ))*K1 + (c32/τ)*K21)*E
        R3 = real(R3+R3')/2
        utqu!(R3, Z) # R3 = Z'*R3*Z
        lyapcs!(Fs, Es, R3; adj=true)
        K31 = R3
        utqu!(K31, Q') # K31 = Q*K31*Q'

        # Update X
        X = X + (m1+m2+m3)*K1 + m2*K21 + m3*K31
        save_state && push!(Xs, X)

        # Update K
        K = (B'*X)*E
        push!(Ks, K)
    end
    save_state || push!(Xs, X)

    return DRESolution(Xs, Ks, tstops)
end
