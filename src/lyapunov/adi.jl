function CommonSolve.solve(
    prob::GALEProblem{T},
    ::ADI;
    nsteps=20,
    rtol=size(prob.A, 1) * eps(),
) where {T <: LDLᵀ}
    @unpack E, A, C = prob
    G, S = C
    ρ(X) = norm((X'X)*S) # Frobenius
    atol = rtol * ρ(G)

    # Compute initial shift parameters
    μ = qshifts(E, A, G)

    # Perform actual ADI
    i = 1
    Ls = Vector{typeof(G)}()
    W = G
    local V, V1, V2 # ADI increments
    local ρW # norm of residual
    while true
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
        F = A' + μ[i]*E
        V = F \ W

        if isreal(μ[i])
            push!(Ls, V)
            W -= 2μ[i] * (E'*V)
            i += 1
        else
            @assert μ[i+1] ≈ conj(μ[i])
            δ = real(μ[i]) / imag(μ[i])
            Vᵣ = real(V)
            Vᵢ = imag(V)
            V′ = Vᵣ + δ*Vᵢ
            V1 = √2 * V′
            V2 = sqrt(2δ^2 + 1) * Vᵢ
            push!(Ls, V1, V2)
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
    @info "ADI done" μ i nsteps residual=ρW atol

    n = size(G, 1)
    L = similar(G, n, sum(_L -> size(_L, 2), Ls))
    col = 0
    for _L in Ls
        ncols = size(_L, 2)
        L[:, col+1:col+ncols] .= _L
        col += ncols
    end
    D = kron(-2*Diagonal(real(μ[1:i])), S)
    return T(L, D)
end

function qshifts(E, A, N)
    Q = orth(N)
    Ẽ = Q'E*Q
    Ã = Q'A*Q
    λ = eigvals(Ã, Ẽ)
    λ₋ = filter(l -> real(l) < 0, λ)
    return λ₋
end

function orth(N)
    λ, N̂ = eigen(Symmetric(N'N); sortby=-)
    ε = count(>(0), λ) * eps() # cf. [Kürschner2016, p. 94]
    d = findlast(>=(ε*λ[1]), λ)
    D̂ = Diagonal(λ[1:d])
    N̂ = N̂[:, 1:d]
    Q = N * (N̂ * (D̂^-0.5))
    #@assert Q'Q ≈ I # FIXME
    return Q
end
