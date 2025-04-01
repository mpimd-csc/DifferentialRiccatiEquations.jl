# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _solve(
    prob::GDREProblem{<:LDLᵀ},
    alg::Ros1;
    dt::Real,
    save_state::Bool,
    observer,
)
    @timeit_debug "callbacks" observe_gdre_start!(observer, prob, alg)

    T = typeof(prob.X0)
    TL = eltype(prob.X0.Ls)
    TD = eltype(prob.X0.Ds)

    @unpack E, A, B, C, tspan = prob
    q = size(C, 1)
    X = prob.X0::T
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Output Trajectories
    Xs = [X]
    save_state && sizehint!(Xs, len)
    alpha, L, D = X
    BᵀLD = adapt(TD, B'L) * D
    alpha == 1 || rmul!(BᵀLD, alpha)
    K = adapt(TL, BᵀLD) * (L'E)
    Ks = [K]
    sizehint!(Ks, len)

    @timeit_debug "callbacks" observe_gdre_step!(observer, tstops[1], X, K)

    inner_alg = @something(alg.inner_alg, ADI())
    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        # Coefficient Matrix of the Lyapunov Equation
        F = lr_update(A - E/(2τ), -1, B, K)

        # Right-hand side:
        G::TL = _hcat(TL, C', E'L)
        S::TD = _dcat(TD, (I(q), (BᵀLD)' * BᵀLD + D/τ))
        R::T = compress!(lowrank(G, S))

        # Update X
        lyap = GALEProblem(E, F, R)
        initial_guess = X
        X = @timeit_debug "$(typeof(inner_alg))" solve(lyap, inner_alg; observer, initial_guess)
        save_state && push!(Xs, X)

        # Update K
        alpha, L, D = X
        BᵀLD = adapt(TD, B'L) * D
        alpha == 1 || rmul!(BᵀLD, alpha)
        K = adapt(TL, BᵀLD) * (L'E)
        push!(Ks, K)

        @timeit_debug "callbacks" observe_gdre_step!(observer, tstops[i], X, K)
    end
    save_state || push!(Xs, X)

    @timeit_debug "callbacks" observe_gdre_done!(observer)

    return DRESolution(Xs, Ks, tstops)
end
