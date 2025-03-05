# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function _zeros(T, m::Int, n::Int)
    X = similar(T, m, n)
    fill!(X, zero(eltype(X)))
end

_zeros(::Type{Matrix{T}}, m::Int, n::Int=m) where {T} = zeros(T, m, n)
_zeros(::Type{<:SparseArrays.SparseMatrixCSC{T}}, m::Int, n::Int=m) where {T} = spzeros(T, m, n)
