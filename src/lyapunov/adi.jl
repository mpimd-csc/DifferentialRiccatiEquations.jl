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
    X::LDLᵀ{TL,TD} = LDLᵀ{TL,TD}(n, 0)
    W::TL = G
    local V, V1, V2 # ADI increments
    local ρW # norm of residual
    while true
        i % 5 == 0 && @debug "ADI" i rank(X) ρW
        # If we exceeded the shift parameters, compute new ones:
        if i > length(μ)
            @debug "Computing new shifts" i
            μ′ = if isreal(μ[i-1])
                qshifts(E, A, V)
            else
                qshifts(E, A, [V1 V2])
            end
            @debug "Obtained $(length(μ′)) new shifts" i
            append!(μ, μ′)
        end

        # Continue with ADI:
        # TODO: Sherman-Morrison-Woodbury
        Y::TD = -2real(μ[i]) * S
        if isreal(μ[i])
            μᵢ = real(μ[i])
            F = A' + μᵢ*E
            V = F \ W

            X += LDLᵀ(V, Y)
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
            V₁ = LDLᵀ(√2 * V′, Y)
            V₂ = LDLᵀ(sqrt(2δ^2 + 1) * Vᵢ, Y)
            X = X + V₁ + V₂
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

    i -= 1 # actual number of ADI steps performed
    @debug "ADI done" i nsteps residual=ρW atol rank(X) extrema(diag(X.D))

    return X
end

function qshifts(E, A, N::AbstractMatrix{<:Real})
    Q = orth(N)
    Ẽ = Q'E*Q
    Ã = Q'A*Q
    λ = eigvals(Ã, Ẽ)
    λ₋ = filter(l -> real(l) < 0, λ)
    return λ₋
end

function orth(N)
    NᵀN = Symmetric(Matrix(N'N))
    λ, N̂ = eigen(NᵀN; sortby=-)
    ε = count(>(0), λ) * eps() # cf. [Kürschner2016, p. 94]
    d = findlast(>=(ε*λ[1]), λ)
    D̂ = Diagonal(λ[1:d])
    N̂ = N̂[:, 1:d]
    Q = N * (N̂ * (D̂^-0.5))
    #@assert Q'Q ≈ I # FIXME
    !(Q'Q ≈ I) && @error "Q not orthonormal" typeof(N) size(N) norm(Q'Q - I)
    return Q
end
