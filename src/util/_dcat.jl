_dcat(::Type{T}, X, Xs...) where {T} = _dcat(T, (X, Xs...))

function _dcat(::Type{T}, Xs) where {T}
    n = sum(X -> size(X, 1), Xs)
    D = _zeros(T, n, n)
    k = 0
    for X in Xs
        l = size(X, 1)
        span = k+1:k+l
        D[span,span] = X
        k += l
    end
    @assert k == n
    return D
end
