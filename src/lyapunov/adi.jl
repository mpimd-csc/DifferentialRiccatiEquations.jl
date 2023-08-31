# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(
    prob::GALEProblem{LDLᵀ{TL,TD}},
    ::ADI;
    initial_guess::LDLᵀ{TL,TD}=zero(prob.C),
    maxiters=100,
    reltol=size(prob.A, 1) * eps(),
    observer=nothing,
) where {TL,TD}
    @unpack E, A, C = prob
    G, S = C
    ρ(X, Y) = norm((X'X)*Y) # Frobenius
    abstol = reltol * ρ(G, S) # use same tolerance as if initial_guess=zero(C)

    # Compute initial shift parameters
    μ::Vector{ComplexF64} = qshifts(E, A, G)

    # Compute initial residual
    X::LDLᵀ{TL,TD} = initial_guess::LDLᵀ{TL,TD}
    R::TL, T::TD = initial_residual = residual(prob, X)::LDLᵀ{TL,TD}
    initial_residual_norm = ρ(R, T)

    # Perform actual ADI
    i = 1
    local V, V₁, V₂ # ADI increments
    local ρR # norm of residual

    observe_gale_start!(observer, prob, ADI(), abstol, reltol)
    observe_gale_metadata!(observer, "ADI shifts", μ)
    observe_gale_step!(observer, 0, X, initial_residual, initial_residual_norm)
    while true
        i % 5 == 0 && @debug "ADI" i rank(X) residual=ρR
        # If we exceeded the shift parameters, compute new ones:
        if i > length(μ)
            @debug "Computing new shifts" i
            μ′ = if isreal(μ[i-1])
                qshifts(E, A, V)
            else
                qshifts(E, A, [V₁ V₂])
            end
            @debug "Obtained $(length(μ′)) new shifts" i
            observe_gale_metadata!(observer, "ADI shifts", μ′)
            append!(μ, μ′)
        end

        # Continue with ADI:
        Y = (-2real(μ[i]) * T)::TD
        if isreal(μ[i])
            μᵢ = real(μ[i])
            F = A' + μᵢ*E
            V = (F \ R)::TL

            X += (V, Y)
            R -= (2μᵢ * (E'*V))::TL
            i += 1
        else
            @assert μ[i+1] ≈ conj(μ[i])
            μᵢ = μ[i]
            F = A' + μᵢ*E
            V = F \ R

            δ = real(μ[i]) / imag(μ[i])
            Vᵣ = real(V)
            Vᵢ = imag(V)
            V′ = Vᵣ + δ*Vᵢ
            V₁ = √2 * V′
            V₂ = sqrt(2δ^2 + 2) * Vᵢ
            X = X + (V₁, Y) + (V₂, Y)
            R -= (4real(μ[i]) * (E'*V′))::TL
            i += 2
        end

        ρR = ρ(R, T)
        observe_gale_step!(observer, i-1, X, LDLᵀ(R, T), ρR)
        ρR <= abstol && break
        if i > maxiters
            observe_gale_failed!(observer)
            @warn "ADI did not converge" residual=ρR abstol maxiters
            break
        end
    end

    _, D = X # run compression, if necessary

    iters = i - 1 # actual number of ADI steps performed
    @debug "ADI done" i=iters maxiters residual=ρR abstol rank(X) rank_initial_guess=rank(initial_guess) rank_rhs=rank(C) rank_residual=size(R)
    observe_gale_done!(observer, iters, X, LDLᵀ(R, T), ρR)

    return X
end

function residual(
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

function qshifts(E, A, N::AbstractMatrix{<:Real})
    Q = orth(N)
    Ẽ = restrict(E, Q)
    Ã = restrict(A, Q)
    λ = eigvals(Ã, Ẽ)
    # TODO: flip values at imaginary axes instead
    λ₋ = filter(l -> real(l) < 0, λ)
    return λ₋
end

orth(N::SparseMatrixCSC) = orth(Matrix(N))

function orth(N::Matrix{T}) where {T}
    if VERSION < v"1.7"
        QR = qr(N, Val(true)) # pivoted
    else
        QR = qr(N, ColumnNorm())
    end
    R = QR.R
    # TODO: Find reference! As of LAPACK 3.1.2 or so,
    # the diagonal of R is sorted with decreasing absolute value,
    # and R is diagonal dominant. Therefore, it may be used to discover the rank.
    # Note that column permutations don't matter for span(N) == span(Q).
    ε = size(N, 1) * eps()
    r = 0
    for outer r in 1:size(R, 1)
        abs(R[r,r]) > ε && continue
        r -= 1
        break
    end
    Q = zeros(T, size(N, 1), r)
    for i in 1:r
        Q[i,i] = 1
    end
    lmul!(QR.Q, Q)
    return Q
end
