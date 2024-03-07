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

lr_update(A::Matrix{T}, α, U, V) where {T} = A + (inv(α)*U)*V
lr_update(A::AbstractSparseMatrix, α, U, V) = LowRankUpdate(A, α, U, V)

Base.eltype(::Type{LowRankUpdate{TA,T,TU,TV}}) where {TA,T,TU,TV} = Base.promote_eltype(TA, T, TU, TV)

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

function Matrix(AUV::LowRankUpdate)
    A, α, U, V = AUV
    A + inv(α) * (U * V)
end

@timeit_debug "Sherman-Morrison-Woodbury" function Base.:(\)(AUV::LowRankUpdate, B::AbstractVecOrMat)
    A, α, U, V = AUV

    n = size(B, 1)
    k = size(B, 2)
    A⁻¹_BU = @timeit_debug "solve (sparse)" begin
        BU = similar(B, n, k + size(U, 2))
        BU[:, 1:k] = B
        BU[:, k+1:end] = U
        A \ BU
    end
    A⁻¹B = if B isa AbstractVector
        @assert k == 1
        @view A⁻¹_BU[:, 1]
    else
        @view A⁻¹_BU[:, 1:k]
    end
    A⁻¹U = @view A⁻¹_BU[:, k+1:end]

    @timeit_debug "solve (dense)" begin
        S = α*I + V * A⁻¹U
        S⁻¹VA⁻¹B = S \ (V * A⁻¹B)
    end

    X = A⁻¹B - A⁻¹U*S⁻¹VA⁻¹B
    return X
end

function Base.:(+)(AUV::LowRankUpdate, E::AbstractMatrix)
    @assert issparse(E)
    A, α, U, V = AUV
    LowRankUpdate(A+E, α, U, V)
end

function Base.:(*)(AUV::LowRankUpdate, X::AbstractVecOrMat)
    size(X, 1) == size(X, 2) && @warn(
        "Multiplying LowRankUpdate by square matrix; memory usage may increase severely",
        dim = size(X, 1),
    )
    A, α, U, V = AUV
    A*X + inv(α)*(U*(V*X))
end
