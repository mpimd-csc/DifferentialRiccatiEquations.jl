"""
    LDLᵀ{TL,TD}(L::TL, D::TD)
    LDLᵀ{TL,TD}(Ls::Vector{TL}, Ds::Vector{TD})

A lazy representation of `L * D * L'` that supports the following functions:

* `+(::LDLᵀ, ::LDLᵀ)` and `+(::LDLᵀ{TL,TD}, ::Tuple{TL,TD})`
* `size`
* `rank` which yields the length of the inner dimension, i.e. `size(D, 1)`
* `zero` which yields a rank 0 representation
* [`concatenate!`](@ref) (expert use only)
* [`compress!`](@ref) (expert use only)

Iterating the structure yields `L::TL` and `D::TD`.
This calls [`compress!`](@ref), if necessary.

For convenience, the structure might be converted to a matrix via `Matrix`.
It is recommended to use this only for testing.
"""
struct LDLᵀ{TL,TD}
    Ls::Vector{TL}
    Ds::Vector{TD}

    LDLᵀ(L::TL, D::TD) where {TL, TD} = new{TL,TD}([L], [D])
    LDLᵀ{TL,TD}(L::TL, D::TD) where {TL, TD} = new{TL,TD}([L], [D])
    LDLᵀ{TL,TD}(L::Vector{TL}, D::Vector{TD}) where {TL, TD} = new{TL,TD}(L, D)
end

# Mainly for testing
function Base.Matrix(X::LDLᵀ)
    @unpack Ls, Ds = X
    L = first(Ls)
    n = size(L, 1)
    M = zeros(eltype(L), n, n)
    for (L, D) in zip(Ls, Ds)
        M .+= L * D * L'
    end
    return M
end

# Destructuring via iteration
function Base.iterate(X::LDLᵀ)
    length(X.Ls) > 1 && compress!(X)
    only(X.Ls), Val(:D)
end
Base.iterate(LD::LDLᵀ, ::Val{:D}) = only(LD.Ds), nothing
Base.iterate(::LDLᵀ, _) = nothing

Base.size(X::LDLᵀ, i) = i <= 2 ? size(first(X.Ls), 1) : 1
Base.size(X::LDLᵀ) = (n = size(X, 1); (n, n))

LinearAlgebra.rank(X::LDLᵀ) = sum(D -> size(D, 1), X.Ds)

function Base.zero(X::LDLᵀ{TL,TD}) where {TL,TD}
    n = size(X, 1)
    L = _zeros(TL, n, 0)
    D = _zeros(TD, 0, 0)
    LDLᵀ{TL,TD}(L, D)
end

function Base.:(+)(Xs::LDLᵀ{TL,TD}...) where {TL,TD}
    Ls = TL[]
    Ds = TD[]
    for X in Xs
        append!(Ls, X.Ls)
        append!(Ds, X.Ds)
    end
    X = LDLᵀ{TL,TD}(Ls, Ds)
    maybe_compress!(X)
end

function Base.:(+)(X::LDLᵀ{TL,TD}, LDs::Tuple{TL,TD}...) where {TL,TD}
    Ls = copy(X.Ls)
    Ds = copy(X.Ds)
    m = length(Ls)
    n = m + length(LDs)
    resize!(Ls, n)
    resize!(Ds, n)
    for (i, (L, D)) in zip(m+1:n, LDs)
        Ls[i] = L
        Ds[i] = D
    end
    Y = LDLᵀ{TL,TD}(Ls, Ds)
    maybe_compress!(Y)
end

function Base.:(-)(X::LDLᵀ{TL,TD}, Y::LDLᵀ{TL,TD}) where {TL,TD}
    Ls = copy(X.Ls)
    Ds = copy(X.Ds)
    L, D = Y
    push!(Ls, L)
    push!(Ds, -D)
    Z = LDLᵀ{TL,TD}(Ls, Ds)
    maybe_compress!(Z)
end

function compression_due(X::LDLᵀ)
    # If there is only one component, it has likely already been compressed:
    length(X.Ls) == 1 && return false
    # Compression is due every couple of modifications:
    # TODO: make this configurable
    length(X.Ls) >= 10 && return true
    # Compression is due if rank is too large:
    # TODO: make this configurable
    n = size(X, 1)
    r = rank(X)
    return r >= 0.5n
end

function maybe_compress!(X::LDLᵀ)
    compression_due(X) || return X
    compress!(X)
end

"""
    concatenate!(X::LDLᵀ)

Concatenate the internal components such that `L` and `D` may be obtained via `L, D = X`.
This function is roughly equivalent to `L = foldl(hcat, X.Ls)` and `D = foldl(dcat, Ds)`,
where `dcat` is pseudo-code for "diagonal concatenation".

This is a somewhat cheap operation.

See also: [`compress!`](@ref)
"""
function concatenate!(X::LDLᵀ{TL,TD}) where {TL,TD}
    @unpack Ls, Ds = X
    @assert length(Ls) == length(Ds)
    length(Ls) == 1 && return X
    n = size(X, 1)
    r = rank(X)
    L::TL = _zeros(TL, n, r)
    D::TD = _zeros(TD, r, r)
    k = 0
    for (_L, _D) in zip(Ls, Ds)
        l = size(_L, 2)
        span = k+1:k+l
        L[:, span] = _L
        D[span, span] = _D
        k += l
    end
    @assert k == r
    resize!(X.Ls, 1)
    resize!(X.Ds, 1)
    X.Ls[1] = L
    X.Ds[1] = D
    return X
end

"""
    compress!(X::LDLᵀ)

Concatenate the internal components and perform a column compression following [^Lang2015].

This is an expensive operation.

See also: [`concatenate!`](@ref)

[^Lang2015]: N Lang, H Mena, and J Saak, "On the benefits of the LDLT factorization for large-scale differential matrix equation solvers" Linear Algebra and its Applications 480 (2015): 44-71. [doi:10.1016/j.laa.2015.04.006](https://doi.org/10.1016/j.laa.2015.04.006)
"""
function compress!(X::LDLᵀ{TL,TD}) where {TL,TD}
    concatenate!(X)
    L = only(X.Ls)
    D = only(X.Ds)
    Q, R, p = qr(L, Val(true)) # pivoting

    ip = invperm(p)
    RΠᵀ = R[:,ip]
    S = Symmetric(RΠᵀ*D*(RΠᵀ)')
    λ, V = eigen(S; sortby=-)

    # only use "large" eigenvalues,
    # cf. [Kürschner2016, p. 94]
    ε = length(λ) * eps()
    λmax = max(1, λ[1])
    r = something(findlast(>=(ε*λmax), λ), 0)

    Vᵣ = @view V[:, 1:r]
    X.Ls[1] = (Q * Vᵣ)::TL
    X.Ds[1] = _diagm(TD, λ[1:r])::TD
    @debug "Compressed LDLᵀ" oldrank=size(D, 1) newrank=r
    return X
end
