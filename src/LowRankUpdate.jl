# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
    LowRankUpdate{TA,T,TU,TV}(A::TA, α::T, U::TU, V::TV)

Lazy representation of `A + inv(α)*U*V` that supports the following functions:

* `\\` via the Sherman-Morrison-Woodbury formula
* `+(::LowRankUpdate, ::AbstractMatrix)` to update `A`
* `adjoint` which returns a `LowRankUpdate`
* `size`

Iterating the structure produces the components `A`, `α`, `U` and `V`.

It is recommended to use [`lr_update`](@ref) to create a suitable
representation of `A + inv(α)*U*V`.
"""
struct LowRankUpdate{TA,T,TU,TV}
    A::TA
    α::T
    U::TU
    V::TV

    LowRankUpdate{T,TA,TU,TV}(A, α, U, V) where {TA,T,TU,TV} = new{TA,T,TU,TV}(A, α, U, V)
    LowRankUpdate(A::TA, α::T, U::TU, V::TV) where {TA,T,TU,TV} = new{TA,T,TU,TV}(A, α, U, V)
end

"""
    lr_update(A::Matrix, α, U, V)
    lr_update(A::AbstractSparseMatrixCSC, α, U, V)

Return a suitable representation of `A + inv(α)*U*V`.
For dense `A`, compute `A + inv(α)*U*V` directly.
For sparse `A`, return a [`LowRankUpdate`](@ref).
"""
lr_update

lr_update(A::Matrix, α, U, V) = A + (inv(α)*U)*V
lr_update(A::AbstractSparseMatrixCSC, α, U, V) = LowRankUpdate(A, α, U, V)

Base.iterate(AUV::LowRankUpdate) = AUV.A, Val(:a)
Base.iterate(AUV::LowRankUpdate, ::Val{:a}) = AUV.α, Val(:U)
Base.iterate(AUV::LowRankUpdate, ::Val{:U}) = AUV.U, Val(:V)
Base.iterate(AUV::LowRankUpdate, ::Val{:V}) = AUV.V, nothing

Base.size(AUV::LowRankUpdate) = size(AUV.A)
Base.size(AUV::LowRankUpdate, i) = size(AUV.A, i)

function Base.adjoint(AUV::LowRankUpdate)
    A, α, U, V = AUV
    LowRankUpdate(A', α', V', U')
end

_factorize(X) = factorize(X)
function _factorize(X::AbstractSparseMatrixCSC)
    F = factorize(X)
    F isa Diagonal || return F
    # If `F` is a `Diagonal`, its diagonal will be a sparse vector.
    # Solving with a dense diagonal has better performance.
    D = Diagonal(Vector(F.diag))
    return D
end

function Base.:(\)(AUV::LowRankUpdate, B::AbstractVecOrMat)
    A, α, U, V = AUV

    FA = _factorize(A)
    A⁻¹B = FA \ B
    A⁻¹U = FA \ U

    S = α*I + V * A⁻¹U
    S⁻¹VA⁻¹B = S \ (V * A⁻¹B)

    X = A⁻¹B - A⁻¹U*S⁻¹VA⁻¹B
    return X
end

function Base.:(+)(AUV::LowRankUpdate, E::AbstractMatrix)
    @assert issparse(E)
    A, α, U, V = AUV
    LowRankUpdate(A+E, α, U, V)
end
