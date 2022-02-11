function CommonSolve.solve(
    prob::GALEProblem{LDLᵀ{TL,TD}},
    ::ADI;
    nsteps=100,
    rtol=size(prob.A, 1) * eps(),
) where {TL,TD}
    @unpack E, A, C = prob
    G, S = C
    ρ(X) = norm((X'X)*S) # Frobenius
    atol = rtol * ρ(G)

    # Compute initial shift parameters
    μ::Vector{ComplexF64} = qshifts(E, A, G)

    # Perform actual ADI
    i = 1
    n = size(G, 1)
    X::LDLᵀ{TL,TD} = zero(C)
    W::TL = G
    local V, V₁, V₂ # ADI increments
    local ρW # norm of residual
    while true
        i % 5 == 0 && @debug "ADI" i rank(X) ρW
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
            V = F \ W

            X += (V, Y)
            W -= 2μ[i] * (E'*V)
            i += 1
        else
            @assert μ[i+1] ≈ conj(μ[i])
            μᵢ = μ[i]
            F = A' + μᵢ*E
            V = F \ W

            δ = real(μ[i]) / imag(μ[i])
            Vᵣ = real(V)
            Vᵢ = imag(V)
            V′ = Vᵣ + δ*Vᵢ
            V₁ = √2 * V′
            V₂ = sqrt(2δ^2 + 1) * Vᵢ
            X = X + (V₁, Y) + (V₂, Y)
            W -= 4real(μ[i]) * (E'*V′)
            i += 2
        end

        ρW = ρ(W)
        ρW <= atol && break
        if i > nsteps
            @warn "ADI did not converge" residual=ρW atol
            break
        end
    end

    _, D = X # run compression, if necessary

    i -= 1 # actual number of ADI steps performed
    @debug "ADI done" i nsteps residual=ρW atol rank(X) extrema(D)

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
    QR = qr(N, Val(true)) # pivoted
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
