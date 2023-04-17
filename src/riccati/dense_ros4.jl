# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _solve(
    prob::GDREProblem{<:Matrix},
    alg::Ros4;
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

    CᵀC = C'C
    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        gF = (τ*(A-B*K)-E)/2
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
        EK1E = E'*K1*E
        EK1B = E'*(K1*B)
        R2 = -τ^2*(EK1B*EK1B')-2*EK1E
        R2 = real(R2+R2')/2
        utqu!(R2, Z) # R2 = Z'*R2*Z
        lyapcs!(Fs, Es, R2; adj=true)
        K21 = R2
        utqu!(K21, Q') # K21 = Q*K21*Q'
        K2 = K21 - K1

        # Solve Lyapunov equation of 3rd stage
        α = (24/25)*τ
        β = (3/25)*τ
        EK2E = E'*K2*E
        EK2B = E'*(K2*B)
        TMP = EK2B*EK1B'
        R3 = (245/25)*EK1E + (36/25)*EK2E - (426/625)*τ^2*(EK1B*EK1B') - β^2*(EK2B*EK2B') - α*β*(TMP+TMP')
        R3 = real(R3+R3')/2
        utqu!(R3, Z) # R3 = Z'*R3*Z
        lyapcs!(Fs, Es, R3; adj=true)
        K31 = R3
        utqu!(K31, Q') # K31 = Q*K31*Q'
        K3 = K31 - (17/25)*K1

        # Solve Lyapunov equation of 4th stage
        R4 = -(981/125)*EK1E-(177/125)*EK2E-(1/5)*E'*K3*E
        R4 = real(R4+R4')/2
        utqu!(R4, Z) # R4 = Z'*R4*Z
        lyapcs!(Fs, Es, R4; adj=true)
        K41 = R4
        utqu!(K41, Q') # K41 = Q*K41*Q'
        K4 = K41 + K3

        # Update X
        X = X + τ*((19/18)*K1 + 0.25*K2 + (25/216)*K3 + (125/216)*K4)
        save_state && push!(Xs, X)

        # Update K
        K = (B'*X)*E
        push!(Ks, K)
    end
    save_state || push!(Xs, X)

    return DRESolution(Xs, Ks, tstops)
end
