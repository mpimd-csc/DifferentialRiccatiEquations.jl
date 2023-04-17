# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _solve(
    prob::GDREProblem{<:Matrix},
    alg::Ros1;
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

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        # Coefficient Matrix of the Lyapunov Equation
        F = (A-B*K) - E/(2τ)
        R = C'*C + K'*K + (1/τ)*E'*X*E

        # Only for safety
        R = real(R+R')/2

        # Update X
        X = lyapc(F', Ed', R)
        save_state && push!(Xs, X)

        # Update K
        K = (B'*X)*E
        push!(Ks, K)
    end
    save_state || push!(Xs, X)

    return DRESolution(Xs, Ks, tstops)
end
