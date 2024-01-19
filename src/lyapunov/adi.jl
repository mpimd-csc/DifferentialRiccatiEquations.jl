# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Compat: @something

function CommonSolve.solve(
    prob::GALEProblem{LDLᵀ{TL,TD}},
    ::ADI;
    initial_guess::Union{Nothing,LDLᵀ{TL,TD}}=nothing,
    maxiters=100,
    reltol=size(prob.A, 1) * eps(),
    abstol=reltol * norm(prob.C), # use same tolerance as if initial_guess=zero(C)
    observer=nothing,
    shifts::Shifts.Strategy=Shifts.Projection(2),
) where {TL,TD}
    @timeit_debug "callbacks" observe_gale_start!(observer, prob, ADI())
    initial_guess = @something initial_guess zero(prob.C)

    @unpack E, A, C = prob

    # Compute initial residual
    X::LDLᵀ{TL,TD} = initial_guess::LDLᵀ{TL,TD}
    R::TL, T::TD = initial_residual = residual(prob, X)::LDLᵀ{TL,TD}
    initial_residual_norm = norm(initial_residual)

    # Initialize shifts
    @timeit_debug "shifts" begin
        shifts = Shifts.init(shifts, prob)
        Shifts.update!(shifts, X, R)
    end

    # Perform actual ADI
    i = 1
    local V, V₁, V₂ # ADI increments
    local ρR # norm of residual

    @timeit_debug "callbacks" observe_gale_step!(observer, 0, X, initial_residual, initial_residual_norm)
    while true
        μ = @timeit_debug "shifts" Shifts.take!(shifts)
        @timeit_debug "callbacks" observe_gale_metadata!(observer, "ADI shifts", μ)

        # Continue with ADI:
        Y = (-2real(μ) * T)::TD
        if isreal(μ)
            μᵢ = real(μ)
            F = A' + μᵢ*E
            @timeit_debug "solve (real)" V = (F \ R)::TL

            X += (V, Y)
            R -= (2μᵢ * (E'*V))::TL
            i += 1

            @timeit_debug "shifts" Shifts.update!(shifts, X, R, V)
        else
            μ_next = @timeit_debug "shifts" Shifts.take!(shifts)
            @assert μ_next ≈ conj(μ)
            @timeit_debug "callbacks" observe_gale_metadata!(observer, "ADI shifts", μ_next)
            μᵢ = μ
            F = A' + μᵢ*E
            @timeit_debug "solve (complex)" V = F \ R

            δ = real(μᵢ) / imag(μᵢ)
            Vᵣ = real(V)
            Vᵢ = imag(V)
            V′ = Vᵣ + δ*Vᵢ
            V₁ = √2 * V′
            V₂ = sqrt(2δ^2 + 2) * Vᵢ
            X = X + (V₁, Y) + (V₂, Y)
            R -= (4real(μ) * (E'*V′))::TL
            i += 2

            @timeit_debug "shifts" Shifts.update!(shifts, X, R, V₁, V₂)
        end

        residual = LDLᵀ(R, T)
        ρR = norm(residual)
        @timeit_debug "callbacks" observe_gale_step!(observer, i-1, X, residual, ρR)
        @debug "ADI" i rank(X) residual=ρR
        ρR <= abstol && break
        if i > maxiters
            @timeit_debug "callbacks" observe_gale_failed!(observer)
            @warn "ADI did not converge" residual=ρR abstol maxiters
            break
        end
    end

    _, D = X # run compression, if necessary

    iters = i - 1 # actual number of ADI steps performed
    @debug "ADI done" i=iters maxiters residual=ρR abstol rank(X) rank_initial_guess=rank(initial_guess) rank_rhs=rank(C) rank_residual=size(R)
    @timeit_debug "callbacks" observe_gale_done!(observer, iters, X, LDLᵀ(R, T), ρR)

    return X
end

@timeit_debug "residual(::GALEProblem, ::LDLᵀ)" function residual(
    prob::GALEProblem{LDLᵀ{TL,TD}},
    val::LDLᵀ{TL,TD},
) where {TL,TD}

    @unpack E, A, C = prob
    G, S = C
    L, D = val
    n_G = size(G, 2)
    n_0 = size(L, 2)
    dim = n_G + 2n_0
    dim == n_G && return C

    R::TL = _hcat(TL, G, E'L, A'L)
    T::TD = _zeros(TD, dim, dim)
    i1 = 1:n_G
    i2 = (1:n_0) .+ n_G
    i3 = i2 .+ n_0
    T[i1, i1] = S
    T[i3, i2] = D
    T[i2, i3] = D

    R̃ = LDLᵀ(R, T)::LDLᵀ{TL,TD}
    compress!(R̃) # unconditionally
end
