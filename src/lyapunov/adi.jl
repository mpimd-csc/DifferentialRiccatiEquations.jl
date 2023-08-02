# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(
    prob::GALEProblem{LDLᵀ{TL,TD}},
    ::ADI;
    maxiters=100,
    reltol=size(prob.A, 1) * eps(),
) where {TL,TD}
    @unpack E, A, C = prob
    G, S = C
    ρ(X) = norm((X'X)*S) # Frobenius
    abstol = reltol * ρ(G)

    # Compute initial shift parameters
    μ::Vector{ComplexF64} = qshifts(E, A, G)

    # Perform actual ADI
    i = 1
    n = size(G, 1)
    X::LDLᵀ{TL,TD} = zero(C)
    R::TL = G # residual
    local V, V₁, V₂ # ADI increments
    local ρR # norm of residual
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
            append!(μ, μ′)
        end

        # Continue with ADI:
        Y::TD = -2real(μ[i]) * S
        if isreal(μ[i])
            μᵢ = real(μ[i])
            F = A' + μᵢ*E
            V = F \ R

            X += (V, Y)
            R -= 2μ[i] * (E'*V)
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
            R -= 4real(μ[i]) * (E'*V′)
            i += 2
        end

        ρR = ρ(R)
        ρR <= abstol && break
        if i > maxiters
            @warn "ADI did not converge" residual=ρR abstol maxiters
            break
        end
    end

    _, D = X # run compression, if necessary

    i -= 1 # actual number of ADI steps performed
    @debug "ADI done" i maxiters residual=ρR abstol rank(X) extrema(D)

    return X
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
