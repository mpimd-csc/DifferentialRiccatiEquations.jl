# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _solve(
    prob::GDREProblem{LDLᵀ{TL,TD}},
    ::Ros1;
    dt::Real,
    save_state::Bool,
    adi_initprev::Bool=true,
    adi_kwargs=NamedTuple(),
    observer,
) where {TL,TD}
    @timeit_debug "callbacks" observe_gdre_start!(observer, prob, Ros1())

    T = LDLᵀ{TL,TD}

    @unpack E, A, B, C, tspan = prob
    q = size(C, 1)
    X = prob.X0::T
    tstops = tspan[1]:dt:tspan[2]
    len = length(tstops)

    # Output Trajectories
    Xs = [X]
    save_state && sizehint!(Xs, len)
    L, D = X
    BᵀLD = adapt(TD, B'L) * D
    K = adapt(TL, BᵀLD) * (L'E)
    Ks = [K]
    sizehint!(Ks, len)

    @timeit_debug "callbacks" observe_gdre_step!(observer, tstops[1], X, K)

    for i in 2:len
        τ = tstops[i-1] - tstops[i]

        # Coefficient Matrix of the Lyapunov Equation
        F = lr_update(A - E/(2τ), -1, B, K)

        # Right-hand side:
        G::TL = _hcat(TL, C', E'L)
        S::TD = _dcat(TD, I(q), (BᵀLD)' * BᵀLD + D/τ)
        R::T = compress!(LDLᵀ(G, S))

        # Update X
        lyap = GALEProblem(E, F, R)
        initial_guess = adi_initprev ? X : nothing
        X = @timeit_debug "ADI" solve(lyap, ADI(; adi_kwargs...); observer, initial_guess)
        save_state && push!(Xs, X)

        # Update K
        L, D = X
        BᵀLD = adapt(TD, B'L) * D
        K = adapt(TL, BᵀLD) * (L'E)
        push!(Ks, K)

        @timeit_debug "callbacks" observe_gdre_step!(observer, tstops[i], X, K)
    end
    save_state || push!(Xs, X)

    @timeit_debug "callbacks" observe_gdre_done!(observer)

    return DRESolution(Xs, Ks, tstops)
end
