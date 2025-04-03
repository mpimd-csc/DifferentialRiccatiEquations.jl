# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Compat: @something

@kwdef mutable struct ADICache{Eltype <: Real, Ztype <: AbstractMatrix, Rtype <: AbstractMatrix, Ttype <:AbstractMatrix}
    prob::GALEProblem
    alg::ADI
    observer
    shifts_oracle
    shifts::Vector{Complex{Eltype}} # shift parameters consumed thus far
    abstol::Real
    last_compression::Int = 0 # number of steps after last `compress!(X)`
    X::LDLᵀ{Eltype,Ztype,Ttype} # current iterate
    increment::LDLᵀ{Eltype,Rtype,Ttype} # last increment before `X += increment`, initially zero
    # Parts of increments:
    V1::Union{Nothing,Rtype} = nothing
    V2::Union{Nothing,Rtype} = nothing
    # Residual factors:
    R::Rtype
    T::Ttype
    residual_norm::Eltype
end

function uses_mixed_precision(::ADICache{Eltype,Ztype,Rtype,Ttype}) where {Eltype,Ztype,Rtype,Ttype}
    !allequal(eltype, (Eltype, Ztype, Rtype, Ttype))
end

residual(cache::ADICache) = lowrank(cache.R, cache.T)

function CommonSolve.init(
    prob::GALEProblem{<:LDLᵀ},
    alg::ADI;
    initial_guess::Union{Nothing,<:LDLᵀ}=nothing,
    initial_residual::Union{Nothing,<:LDLᵀ}=nothing,
    abstol=nothing,
    observer=nothing,
)
    @timeit_debug "callbacks" observe_gale_start!(observer, prob, alg)
    @unpack E, A, C = prob

    # Compute initial residual
    if alg.ignore_initial_guess || initial_guess === nothing
        initial_guess = zero(C)
    end
    if initial_residual === nothing
        initial_residual = residual(prob, initial_guess)
    end
    X = initial_guess
    alpha, R, T = initial_residual
    residual_norm = norm(initial_residual)::eltype(X)
    @assert alpha == 1
    @assert eltype(X.Ds) == eltype(initial_residual.Ds)

    # Initialize shifts
    @timeit_debug "shifts" begin
        shifts_oracle = Shifts.init(alg.shifts, prob)
        Shifts.update!(shifts_oracle, X, R)
        shifts = Complex{eltype(X)}[]
        sizehint!(shifts, alg.maxiters)
    end

    # Initialize tolerances
    reltol = @something(alg.reltol, size(A, 1) * eps(eltype(X)))
    abstol = @something(abstol, alg.abstol, reltol * norm(C)) # use same tolerance as if initial_guess=zero(C)

    @debug "ADI start" reltol abstol residual=residual_norm rank(X) rank(initial_residual)
    @timeit_debug "callbacks" observe_gale_step!(observer, 0, X, initial_residual, residual_norm)

    increment = zero(initial_residual)
    ADICache(; prob, alg, abstol, observer, shifts_oracle, shifts, R, T, X, increment, residual_norm)
end

function CommonSolve.solve!(cache::ADICache)
    # Perform actual ADI
    while !isdone(cache)
        step!(cache)
    end

    # Run compression, if necessary:
    if cache.last_compression > 0
        compress!(cache)
    end

    (; X, abstol, residual_norm) = cache
    (; maxiters) = cache.alg
    iters = length(cache.shifts) # actual number of ADI steps performed
    @debug "ADI done" i=iters maxiters residual=residual_norm abstol rank(X) rank_rhs=rank(cache.prob.C) rank_residual=rank(residual(cache))
    @timeit_debug "callbacks" observe_gale_done!(cache.observer, iters, X, residual(cache), residual_norm)

    return X
end

function Base.iterate(cache::ADICache, done=false)
    done && return nothing
    step!(cache)
    return cache, isdone(cache)
end

function CommonSolve.step!(cache::ADICache)
    (; alg, abstol, observer, shifts_oracle) = cache
    (; maxiters) = alg

    μ = @timeit_debug "shifts" Shifts.take!(shifts_oracle)
    push!(cache.shifts, μ)
    @timeit_debug "callbacks" observe_gale_metadata!(observer, "ADI shifts", μ)

    if isreal(μ)
        perform_single_step!(cache, real(μ))
    else
        perform_double_step!(cache, μ)
    end

    if cache.last_compression >= alg.compression_interval
        compress!(cache)
    end

    X = cache.X
    res = residual(cache)
    res_norm = cache.residual_norm = norm(res)
    i = length(cache.shifts)
    @timeit_debug "callbacks" observe_gale_step!(observer, i, X, res, res_norm)
    @debug "ADI" i abstol residual=res_norm rank(X) compressed=iszero(cache.last_compression)

    res_norm <= abstol && return nothing
    i <= maxiters && return nothing

    @timeit_debug "callbacks" observe_gale_failed!(observer)
    @warn "ADI did not converge" residual=res_norm abstol maxiters
    return nothing
end

function isdone(cache::ADICache)
    # Did we converge?
    cache.residual_norm <= cache.abstol && return true

    # Did the iteration collapse?
    # I only observed this for mixed-precision computations thus far.
    niters = length(cache.shifts)
    niters > 0 && iszero(cache.increment) && return true

    # Did we exceed the maximum number of iterations?
    niters > cache.alg.maxiters
end

function compress!(cache::ADICache)
    compress!(cache.X)
    cache.last_compression = 0
    return nothing
end

function perform_single_step!(cache::ADICache, μ)
    (; prob, alg, R, T) = cache
    (; E, A) = prob
    Rtype = typeof(cache.R)

    # Compute increment:
    F = A' + (μ*E)'
    @timeit_debug "solve (real)" begin
        inner_prob = BlockLinearProblem(F, R)
        V = solve(inner_prob, alg.inner_alg)::Rtype
    end
    if uses_mixed_precision(cache) && iszero(V)
        @warn "Increment is zero"
        cache.increment = zero(lowrank(R, T))
        return nothing
    end
    increment = -2real(μ) * lowrank(V, T)
    cache.increment = increment::typeof(cache.increment)

    # Update residual:
    # R -= (2μ * (E'*V))::Rtype
    mul!(R, E', V, -2μ, true)

    # Update solution:
    cache.X += cache.increment
    cache.last_compression += 1

    @timeit_debug "shifts" Shifts.update!(cache.shifts_oracle, cache.X, R, V)
    return nothing
end

function perform_double_step!(cache::ADICache, μ)
    (; prob, alg, R, T) = cache
    (; E, A) = prob
    Rtype = typeof(cache.R)
    Vᵣ::Rtype = cache.V1 = @something(cache.V1, similar(R))::Rtype
    Vᵢ::Rtype = cache.V2 = @something(cache.V2, similar(R))::Rtype

    μ_next = @timeit_debug "shifts" Shifts.take!(cache.shifts_oracle)
    @assert μ_next ≈ conj(μ)
    push!(cache.shifts, μ_next)
    @timeit_debug "callbacks" observe_gale_metadata!(cache.observer, "ADI shifts", μ_next)

    # Compute increment:
    F = A' + (conj(μ)*E)'
    @timeit_debug "solve (complex)" begin
        inner_prob = BlockLinearProblem(F, R)
        V = solve(inner_prob, alg.inner_alg)
    end
    if iszero(V)
        @warn "Increment is zero"
        cache.increment = zero(lowrank(R, T))
        return nothing
    end
    δ = real(μ) / imag(μ)
    @. Vᵣ = real(V)
    @. Vᵢ = imag(V)
    # The outer factors V₁ and V₂ must allocate, because they are aliased by X:
    el = eltype(Rtype)
    V₁ = (@. el(√2) * Vᵣ + el(√2 * δ) * Vᵢ)::Rtype
    V₂ = (el(sqrt(2δ^2 + 2)) * Vᵢ)::Rtype
    increment = -2real(μ) * (lowrank(V₁, T) + lowrank(V₂, T))
    cache.increment = increment::typeof(cache.increment)

    # Update residual:
    # R -= (2√2real(μ) * (E'*V₁))::Rtype
    mul!(R, E', V₁, -2√2 * real(μ), true)

    # Update solution:
    cache.X += cache.increment
    cache.last_compression += 2

    @timeit_debug "shifts" Shifts.update!(cache.shifts_oracle, cache.X, R, V₁, V₂)
    return nothing
end
