# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

_hcat(::Type{T}, X, Xs...) where {T} = _hcat(T, (X, Xs...))

function _hcat(::Type{T}, Xs) where {T}
    m = size(first(Xs), 1)
    n = sum(X -> size(X, 2), Xs)
    L = _zeros(T, m, n)
    k = 0
    for X in Xs
        l = size(X, 2)
        span = k+1:k+l
        L[:,span] = X
        k += l
    end
    @assert k == n
    return L
end
