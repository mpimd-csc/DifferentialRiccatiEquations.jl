# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
    LDLᵀ{TL,TD}(L::TL, D::TD)
    LDLᵀ{TL,TD}(Ls::Vector{TL}, Ds::Vector{TD})

A lazy representation of `L * D * L'` that supports the following functions:

* `+(::LDLᵀ, ::LDLᵀ)` and `+(::LDLᵀ{TL,TD}, ::Tuple{TL,TD})`
* `*(::Real, ::LDLᵀ)`
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

Base.eltype(::Type{LDLᵀ{TL,TD}}) where {TL,TD} = promote_type(eltype(TL), eltype(TD))

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

"""
    norm(::LDLᵀ)

Compute the Frobenius norm of a LDLᵀ factorization.
The technique is similar to the one described in

> Benner, Li, Penzl. Numerical solution of large-scale Lyapunov equations,
> Riccati equations, and linear-quadratic optimal control problems.
> Numerical Linear Algebra with Applications 2008. DOI: 10.1002/nla.622
"""
@timeit_debug "norm(::LDLᵀ)" function LinearAlgebra.norm(X::LDLᵀ)
    # Decompose while not triggering compression.
    concatenate!(X)
    L = only(X.Ls)
    D = only(X.Ds)
    # TODO: use specialized TSQR ("tall and skinny QR") algorithm.
    # TODO: evaluate whether `compress!` could share any code with `norm`.
    if VERSION < v"1.7"
        _, R = qr(L, Val(false)) # no pivoting
    else
        _, R = qr(L, NoPivot())
    end
    # The Q operator of the QR decomposition does not alter the Frobenius norm.
    # It may therefore be omitted from the matrix inside the norm.
    norm(R * D * R')
end

LinearAlgebra.rank(X::LDLᵀ) = sum(D -> size(D, 1), X.Ds)

function Base.zero(X::LDLᵀ{TL,TD}) where {TL,TD}
    n = size(X, 1)
    L = _zeros(TL, n, 0)
    D = _zeros(TD, 0, 0)
    LDLᵀ{TL,TD}(L, D)
end

function Base.:(+)(Xs::LDLᵀ{TL,TD}...) where {TL,TD}
    if !allequal(size(X, 1) for X in Xs)
        dims = unique(size(X, 1) for X in Xs)
        throw(DimensionMismatch("outer dimensions must match, got $dims instead"))
    end
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
    if !all(==(size(X, 1)), size(Y[1], 1) for Y in LDs)
        dims = [size(Y[1], 1) for Y in LDs]
        pushfirst!(dims, size(X, 1))
        unique!(dims)
        throw(DimensionMismatch("outer dimensions must match, got $dims instead"))
    end
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
    if size(X, 1) != size(Y, 1)
        d1 = size(X, 1)
        d2 = size(Y, 1)
        throw(DimensionMismatch("outer dimensions must match, got $d1 and $d2 instead"))
    end
    Ls = copy(X.Ls)
    Ds = copy(X.Ds)
    L, D = Y
    push!(Ls, L)
    push!(Ds, -D)
    Z = LDLᵀ{TL,TD}(Ls, Ds)
    maybe_compress!(Z)
end

# TODO: Make this more efficient by storing the scalar as a field of LDLᵀ.
function Base.:(*)(α::Real, X::LDLᵀ)
    L, D = X
    LDLᵀ(L, α*D)
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
@timeit_debug "concatenate!(::LDLᵀ)" function concatenate!(X::LDLᵀ{TL,TD}) where {TL,TD}
    @unpack Ls, Ds = X
    @assert length(Ls) == length(Ds)
    length(Ls) == 1 && return X
    L = _hcat(TL, Ls)
    D = _dcat(TD, Ds)
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
@timeit_debug "compress!(::LDLᵀ)" function compress!(X::LDLᵀ{TL,TD}) where {TL,TD}
    concatenate!(X)
    L = only(X.Ls)
    D = only(X.Ds)
    @timeit_debug "QR" if VERSION < v"1.7"
        Q, R, p = qr(L, Val(true)) # pivoting
    else
        Q, R, p = qr(L, ColumnNorm())
    end

    ip = invperm(p)
    RΠᵀ = R[:,ip]
    S = Symmetric(RΠᵀ*D*(RΠᵀ)')
    λ, V = @timeit_debug "Eigen" eigen(S; sortby = x -> -abs(x))

    # only use "large" eigenvalues,
    # cf. [Kürschner2016, p. 94]
    # (modified to retain negative ones)
    ε = max(1, abs(λ[1])) * length(λ) * eps()
    r = something(findlast(l -> abs(l) >= ε, λ), 0)

    @debug "compress!(::LDLᵀ)" λ[1] λ[end] count(>(ε), λ) count(<(-ε), λ) oldrank=size(D,1) newrank=r

    Vᵣ = @view V[:, 1:r]
    X.Ls[1] = (Q * Vᵣ)::TL
    X.Ds[1] = _diagm(TD, λ[1:r])::TD
    return X
end
