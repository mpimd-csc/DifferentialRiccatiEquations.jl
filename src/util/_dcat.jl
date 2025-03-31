# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _dcat(::Type{T}, Xs) where {T}
    alphas = Iterators.repeated(one(eltype(T)))
    _dcat(T, Xs, alphas)
end

function _dcat(::Type{T}, Xs, alphas) where {T}
    n = sum(X -> size(X, 1), Xs)
    D = _zeros(T, n, n)
    DD = parent(D)
    k = 0
    for (X, alpha) in zip(Xs, alphas)
        l = size(X, 1)
        span = k+1:k+l
        X = adapt(get_backend(D), X)
        @. DD[span,span] = X * alpha
        k += l
    end
    @assert k == n
    return D
end
