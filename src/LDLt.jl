# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Compat: allequal

"""
    lowrank(L, D)::LDLᵀ

A lazy representation of `alpha * L * D * L'` where `alpha::Real` is initially one,
that supports the following functions:

* `+(::LDLᵀ, ::LDLᵀ)`
* `*(::Real, ::LDLᵀ)`
* `eltype(::LDLᵀ)` which yields `Base.promote_eltype(L, D)` (same as `typeof(alpha)`)
* `size`
* `rank` which yields the length of the inner dimension, i.e. `size(L, 2)`
* `zero` which yields a rank 0 representation
* [`concatenate!`](@ref) (expert use only)
* [`compress!`](@ref) (expert use only)

Iterating the structure yields `alpha`, `L` and `D`.
This calls [`compress!`](@ref), if necessary.

For convenience, the structure might be converted to a matrix via `Matrix`.
It is recommended to use this only for testing.
"""
function lowrank(L, D=I)
    T = Base.promote_eltype(L, D)
    LDLᵀ{T, typeof(L), typeof(D)}([one(T)], [L], [D])
end

struct LDLᵀ{T,TL,TD}
    alphas::Vector{T}
    Ls::Vector{TL}
    Ds::Vector{TD}
end

Base.:(==)(X::LDLᵀ, Y::LDLᵀ) = X.alphas == Y.alphas && X.Ls == Y.Ls && X.Ds == Y.Ds

Base.eltype(::Type{<:LDLᵀ{T}}) where {T} = T

# Mainly for testing
function Base.Matrix(X::LDLᵀ)
    @unpack alphas, Ls, Ds = X
    L = first(Ls)
    n = size(L, 1)
    M = zeros(eltype(X), n, n)
    for (a, L, D) in zip(alphas, Ls, Ds)
        M .+= L * (a * D) * L'
    end
    return M
end

# Destructuring via iteration
function Base.iterate(X::LDLᵀ)
    length(X.Ls) > 1 && compress!(X)
    only(X.alphas), Val(:L)
end
Base.iterate(X::LDLᵀ, ::Val{:L}) = only(X.Ls), Val(:D)
Base.iterate(X::LDLᵀ, ::Val{:D}) = only(X.Ds), nothing
Base.iterate(::LDLᵀ, _) = nothing

Base.size(X::LDLᵀ, i) = i <= 2 ? size(first(X.Ls), 1) : 1
Base.size(X::LDLᵀ) = (n = size(X, 1); (n, n))

"""
    norm(::LDLᵀ)

Compute the Frobenius norm of a LDLᵀ factorization.
The technique is similar to the one described in

> Benner, Li, Penzl. Numerical solution of large-scale Lyapunov equations,
> Riccati equations, and linear-quadratic optimal control problems.
> Numerical Linear Algebra with Applications 2008. DOI: 10.1002/nla.622

See also: [`orthf`](@ref)
"""
@timeit_debug "norm(::LDLᵀ)" function LinearAlgebra.norm(X::LDLᵀ)
    # Decompose while not triggering compression.
    concatenate!(X)
    a = only(X.alphas)
    L = only(X.Ls)
    D = only(X.Ds)
    # TODO: evaluate whether `compress!` could share any code with `norm`.
    _, R = orthf(L)
    # The Q operator of the orth-plus-square decomposition does not alter the Frobenius norm.
    # It may therefore be omitted from the matrix inside the norm.
    R = adapt(typeof(D), R)
    abs(a) * norm(restrict(D, R'))
end

# The inner factors may be of type UniformScaling, i.e., they may not have a size.
# Therefore, query the outer factors instead:
LinearAlgebra.rank(X::LDLᵀ) = sum(L -> size(L, 2), X.Ls)

Base.iszero(X::LDLᵀ) = all(iszero, X.alphas) || rank(X) == 0

function Base.zero(X::LDLᵀ)
    n = size(X, 1)
    L = _zeros(eltype(X.Ls), n, 0)
    D = _zeros(eltype(X.Ds), 0, 0)
    lowrank(L, D)::typeof(X)
end

function Base.convert(::Type{LDLᵀ{T,TL,TD}}, X::LDLᵀ) where {T,TL,TD}
    @debug "convert(::Type{<:LDLᵀ}, ::LDLᵀ)" src=typeof(X) dst=LDLᵀ{T,TL,TD}
    alphas = convert(Vector{T}, X.alphas)
    Ls = convert(Vector{TL}, X.Ls)
    Ds = convert(Vector{TD}, X.Ds)
    LDLᵀ{T,TL,TD}(alphas, Ls, Ds)
end

function Base.:(+)(X1::LDLᵀ, X2::LDLᵀ)
    if (n1 = size(X1, 1)) != (n2 = size(X2, 1))
        throw(DimensionMismatch("outer dimensions must match, got $n1 and $n2 instead"))
    end
    iszero(X1) && return X2
    iszero(X2) && return X1
    if typeof(X1) != typeof(X2)
        @warn "Calling +(::LDLᵀ, ::LDLᵀ) with different types; converting right argument" src=typeof(X2) dst=typeof(X1) maxlog=1
    end
    alphas = copy(X1.alphas)
    Ls = copy(X1.Ls)
    Ds = copy(X1.Ds)
    append!(alphas, X2.alphas)
    append!(Ls, X2.Ls)
    append!(Ds, X2.Ds)
    X = typeof(X1)(alphas, Ls, Ds)
    return X
end

function Base.:(-)(X::LDLᵀ)
    eltype(X.Ds) == UniformScaling{Bool} && throw(ArgumentError("Operation not supported"))
    typeof(X)(-X.alphas, X.Ls, X.Ds)
end
Base.:(-)(X::LDLᵀ, Y::LDLᵀ) = X + (-Y)

function Base.:(*)(alpha::Real, X::LDLᵀ)
    alphas = alpha * X.alphas
    typeof(X)(alphas, X.Ls, X.Ds)
end

"""
    concatenate!(X::LDLᵀ)

Concatenate the internal components such that `alpha`, `L` and `D` may be obtained via `alpha, L, D = X`.
This function is roughly equivalent to `L = foldl(hcat, X.Ls)` and `D = foldl(dcat, Ds)`,
where `dcat` is pseudo-code for "diagonal concatenation".

This is a somewhat cheap operation.

See also: [`compress!`](@ref)
"""
@timeit_debug "concatenate!(::LDLᵀ)" function concatenate!(X::LDLᵀ)
    length(X.alphas) == 1 && return X

    TL = eltype(X.Ls)
    TD = eltype(X.Ds)
    @unpack alphas, Ls, Ds = X
    @assert length(alphas) == length(Ls) == length(Ds)
    length(Ls) == 1 && return X
    L = _hcat(TL, Ls)
    D = _dcat(TD, X.Ds, X.alphas)
    resize!(X.alphas, 1)
    resize!(X.Ls, 1)
    resize!(X.Ds, 1)
    X.alphas[1] = one(eltype(X))
    X.Ls[1] = L
    X.Ds[1] = D
    return X
end

"""
    compress!(X::LDLᵀ)

Concatenate the internal components and perform a column compression following [^Lang2015].

This is an expensive operation.

See also: [`concatenate!`](@ref), [`orthf`](@ref)

[^Lang2015]: N Lang, H Mena, and J Saak, "On the benefits of the LDLT factorization for large-scale differential matrix equation solvers" Linear Algebra and its Applications 480 (2015): 44-71. [doi:10.1016/j.laa.2015.04.006](https://doi.org/10.1016/j.laa.2015.04.006)
"""
@timeit_debug "compress!(::LDLᵀ)" function compress!(X::LDLᵀ)
    concatenate!(X)
    L = only(X.Ls)
    D = only(X.Ds)
    TL = typeof(L)
    TD = typeof(D)

    Q, R = @timeit_debug "orthf" orthf(L)
    R = adapt(TD, R)
    S = Symmetric(restrict(D, R'))
    λ, V = @timeit_debug "eigen" eigen(S)

    ε = 100 * maximum(abs, λ) * eps()
    ids = findall(l -> abs(l) >= ε, λ)
    @debug "compress!(::LDLᵀ)" extrema(λ) count(>(ε), λ) count(<(-ε), λ) oldrank=length(λ) newrank=length(ids)

    Vᵣ = V[:, ids]
    Vᵣ = adapt(TL, Vᵣ)
    X.Ls[1] = (Q * Vᵣ)::TL
    X.Ds[1] = _diagm(TD, λ[ids])::TD
    return X
end

"""
    orthf(L) -> Q, R

Compute economy-size factorization `L ≈ Q * R` with orthogonal `Q` and square `R`.

Default: pivoted QR decomposition

This is an internal helper routine within [`compress!`](@ref) and `norm(::LDLᵀ)`,
which is applied to the outer factor of a low-rank factorization.
"""
function orthf(L)
    # TODO: use specialized TSQR ("tall and skinny QR") algorithm.

    pivoted = VERSION < v"1.7" ? Val(true) : ColumnNorm()
    Q, R, p = qr(L, pivoted)
    ip = invperm(p)
    RΠᵀ = R[:, ip]
    Q, RΠᵀ
end
