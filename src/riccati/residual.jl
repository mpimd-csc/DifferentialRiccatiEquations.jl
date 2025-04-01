
# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Compat: @something

@timeit_debug "residual(::GAREProblem, ::LDLᵀ)" function residual(
    prob::GAREProblem,
    X::LDLᵀ;
    AᵀL = nothing,
    EᵀL = nothing,
    BᵀLD = nothing,
    DLᵀGLD = nothing,
)
    @unpack E, A, Q, G = prob
    iszero(X) && return deepcopy(Q)

    gamma, Cᵀ, S = Q
    beta, B, R⁻¹ = G
    alpha, L, D = X
    h = size(Cᵀ, 2)
    zₖ = size(L, 2)
    dim = h + 2zₖ
    @debug "Assembling ARE residual" h zₖ

    # Compute optional inputs
    AᵀL = @something(AᵀL, A'L)
    EᵀL = @something(EᵀL, E'L)
    if DLᵀGLD === nothing
        if BᵀLD === nothing
            BᵀLD = (B'L)*D
            alpha * beta == 1 || rmul!(BᵀLD, alpha * beta)
        end
        DLᵀGLD = (BᵀLD)' * R⁻¹ * BᵀLD
    end

    # Compute residual following Benner, Li, Penzl (2008)
    R = [Cᵀ AᵀL EᵀL]
    T = zeros(eltype(X), dim, dim)
    b1 = 1:h
    b2 = h+1:h+zₖ
    b3 = b2 .+ zₖ
    T11 = view(T, b1, b1)
    T23 = view(T, b2, b3)
    T32 = view(T, b3, b2)
    T33 = view(T, b3, b3)
    mul!(T11, gamma, S)
    mul!(T23, alpha, D)
    copyto!(T32, T23)
    mul!(T33, -one(eltype(DLᵀGLD)), DLᵀGLD)

    compress!(lowrank(R, T))
end

@timeit_debug "residual(::GAREProblem, ::Matrix)" function residual(
    prob::GAREProblem,
    X::Matrix;
)

    @unpack E, A, G, Q = prob
    alpha, B, D = G
    BᵀXE = (B' * X) * E
    res = Matrix(Q)
    res += A' * X * E
    res += E' * X * A
    res -= (BᵀXE)' * (alpha * D) * BᵀXE
end
