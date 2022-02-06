#TODO: add parameter for compression
struct LDLᵀ{TL,TD}
    L::TL
    D::TD

    LDLᵀ(L::TL, D::TD) where {TL, TD} = new{TL,TD}(L, D)
    LDLᵀ{TL,TD}(L::TL, D::TD) where {TL, TD} = new{TL,TD}(L, D)
end

# Mainly for testing
Base.Matrix(X::LDLᵀ) = X.L * X.D * X.L'

# Destructuring via iteration
Base.iterate(LD::LDLᵀ) = LD.L, :L
Base.iterate(LD::LDLᵀ, _) = LD.D, nothing

LinearAlgebra.rank(X::LDLᵀ) = size(X.L, 2)
Base.size(X::LDLᵀ) = (n = size(X.L, 1); (n, n))

function Base.:(+)(Xs::LDLᵀ{TL,TD}...) where {TL,TD}
    _L, _D = Xs[1]
    n = size(_L, 1)
    K = sum(size(X.L, 2) for X in Xs)

    # Collect L and D:
    L = similar(_L, n, K)
    D = similar(_D, K, K)
    c = 0
    for X in Xs
        # Collect L:
        k = size(X.L, 2)
        cols = c+1:c+k
        L[:,cols] = X.L
        # Collect D:
        D[cols,cols] = X.D
        c += k
    end

    X = LDLᵀ{TL,TD}(L, D)
    return compress(X)
end

function Base.:(-)(X::LDLᵀ{TL,TD}, Y::LDLᵀ{TL,TD}) where {TL,TD}
    A, B = X
    C, D = Y

    # outer factor:
    E = [A C]

    # inner factor:
    k = size(B, 1)
    K = k + size(D, 1)
    F = similar(B, K, K)
    F[1:k,1:k] = B
    F[k+1:K,k+1:K] = -D

    Z = LDLᵀ(E, F)
    return compress(Z)
end

# TODO: inplace compression
function compress(X::LDLᵀ{TL,TD}) where {TL,TD}
    L, D = X
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
    Λᵣ = Diagonal(λ[1:r]) # FIXME: Do I really want to change the type here?
    Lᵣ = Q * Vᵣ
    return LDLᵀ(Lᵣ, Λᵣ)
end
