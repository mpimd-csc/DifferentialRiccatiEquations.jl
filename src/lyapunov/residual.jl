# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

@timeit_debug "residual(::GALEProblem, ::LDLᵀ)" function residual(
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

@timeit_debug "residual(::GALEProblem, ::Matrix)" function residual(
    prob::GALEProblem,
    X::Matrix;
)

    @unpack E, A, C = prob
    res = Matrix(C)
    res += A' * X * E
    res += E' * X * A
end