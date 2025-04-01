# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

@timeit_debug "residual(::GALEProblem, ::LDLᵀ)" function residual(
    prob::GALEProblem{<:LDLᵀ},
    val::LDLᵀ,
)

    @unpack E, A, C = prob
    iszero(val) && return deepcopy(C)

    alpha, G, S = C
    beta, L, D = val
    n_G = size(G, 2)
    n_0 = size(L, 2)
    dim = n_G + 2n_0

    TL = eltype(val.Ls)
    R::TL = _hcat(TL, G, E'L, A'L)
    T = zeros(eltype(val), dim, dim)
    i1 = 1:n_G
    i2 = (1:n_0) .+ n_G
    i3 = i2 .+ n_0
    T11 = view(T, i1, i1)
    T23 = view(T, i2, i3)
    T32 = view(T, i3, i2)
    mul!(T11, alpha, S)
    mul!(T23, beta, D)
    copyto!(T32, T23)

    compress!(lowrank(R, T))
end

@timeit_debug "residual(::GALEProblem, ::Matrix)" function residual(
    prob::GALEProblem,
    X::Matrix;
)

    @unpack E, A, C = prob
    res = Matrix(C)
    res += A' * X * E
    res += E' * X * A
end
