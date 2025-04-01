# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Compat: @something

function CommonSolve.solve(
    prob::GALEProblem{TX},
    alg::ADI;
    initial_guess::Union{Nothing,TX}=nothing,
    abstol=nothing,
    observer=nothing,
) where {TX <: LDLᵀ}
    @timeit_debug "callbacks" observe_gale_start!(observer, prob, alg)
    @unpack E, A, C = prob
    reltol = @something(alg.reltol, size(A, 1) * eps())
    abstol = @something(abstol, alg.abstol, reltol * norm(C)) # use same tolerance as if initial_guess=zero(C)

    # Compute initial residual
    if alg.ignore_initial_guess || initial_guess === nothing
        initial_guess = zero(C)
    end
    TL = eltype(C.Ls)
    TD = eltype(C.Ds)
    X::TX = initial_guess::TX
    alpha, R::TL, T::TD = initial_residual = residual(prob, X)
    @assert alpha == 1
    initial_residual_norm = norm(initial_residual)

    # Initialize shifts
    @timeit_debug "shifts" begin
        shifts = Shifts.init(alg.shifts, prob)
        Shifts.update!(shifts, X, R)
    end

    # Perform actual ADI
    @unpack inner_alg, maxiters = alg
    i = 1
    last_compression = 0
    local V, V₁, V₂ # ADI increments
    local ρR # norm of residual

    @timeit_debug "callbacks" observe_gale_step!(observer, 0, X, initial_residual, initial_residual_norm)
    while true
        μ = @timeit_debug "shifts" Shifts.take!(shifts)
        @timeit_debug "callbacks" observe_gale_metadata!(observer, "ADI shifts", μ)

        # Continue with ADI:
        if isreal(μ)
            μᵢ = real(μ)
            F = A' + (μᵢ*E)'
            @timeit_debug "solve (real)" begin
                inner_prob = BlockLinearProblem(F, R)
                V = solve(inner_prob, inner_alg)::TL
            end

            X -= 2real(μ) * lowrank(V, T)
            mul!(R, E', V, -2μᵢ, true) # R -= (2μᵢ * (E'*V))::TL
            i += 1
            last_compression += 1

            @timeit_debug "shifts" Shifts.update!(shifts, X, R, V)
        else
            μ_next = @timeit_debug "shifts" Shifts.take!(shifts)
            @assert μ_next ≈ conj(μ)
            @timeit_debug "callbacks" observe_gale_metadata!(observer, "ADI shifts", μ_next)
            μᵢ = μ
            F = A' + (conj(μᵢ)*E)'
            @timeit_debug "solve (complex)" begin
                inner_prob = BlockLinearProblem(F, R)
                V = solve(inner_prob, inner_alg)
            end

            δ = real(μᵢ) / imag(μᵢ)
            Vᵣ = real(V)
            Vᵢ = imag(V)
            V′ = Vᵣ + δ*Vᵢ
            V₁ = √2 * V′
            V₂ = sqrt(2δ^2 + 2) * Vᵢ
            X -= 2real(μ) * (lowrank(V₁, T) + lowrank(V₂, T))
            mul!(R, E', V′, -4real(μ), true) # R -= (4real(μ) * (E'*V′))::TL
            i += 2
            last_compression += 2

            @timeit_debug "shifts" Shifts.update!(shifts, X, R, V₁, V₂)
        end

        if last_compression >= alg.compression_interval
            compress!(X)
            last_compression = 0
        end

        residual = lowrank(R, T)
        ρR = norm(residual)
        @timeit_debug "callbacks" observe_gale_step!(observer, i-1, X, residual, ρR)
        @debug "ADI" i reltol abstol residual=ρR rank(X) compressed=(last_compression==0)
        ρR <= abstol && break
        if i > maxiters
            @timeit_debug "callbacks" observe_gale_failed!(observer)
            @warn "ADI did not converge" residual=ρR abstol maxiters
            break
        end
    end

    # Run compression, if necessary:
    if last_compression > 0
        compress!(X)
    end

    iters = i - 1 # actual number of ADI steps performed
    @debug "ADI done" i=iters maxiters residual=ρR abstol rank(X) rank_initial_guess=rank(initial_guess) rank_rhs=rank(C) rank_residual=size(R)
    @timeit_debug "callbacks" observe_gale_done!(observer, iters, X, lowrank(R, T), ρR)

    return X
end
