
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
    gamma, Cᵀ, _ = Q
    beta, B, _ = G
    alpha, L, D = X
    h = size(Cᵀ, 2)
    zₖ = size(L, 2)
    dim = h + 2zₖ
    dim == h && return deepcopy(Q)
    @debug "Assembling ARE residual" h zₖ

    # Compute optional inputs
    AᵀL = @something(AᵀL, A'L)
    EᵀL = @something(EᵀL, E'L)
    if DLᵀGLD === nothing
        if BᵀLD === nothing
            BᵀLD = (B'L)*D
            alpha * beta == 1 || rmul!(BᵀLD, alpha * beta)
        end
        DLᵀGLD = (BᵀLD)'BᵀLD
    end

    # Compute residual following Benner, Li, Penzl (2008)
    R = [Cᵀ AᵀL EᵀL]
    T = zeros(dim, dim)
    b1 = 1:h
    b2 = h+1:h+zₖ
    b3 = b2 .+ zₖ
    for i in b1
        T[i, i] = gamma
    end
    T[b2, b3] .= T[b3, b2] .= alpha * D
    T[b3, b3] .= -DLᵀGLD

    lowrank(R, T)
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
    res += (BᵀXE)' * (alpha * D) * BᵀXE
end
