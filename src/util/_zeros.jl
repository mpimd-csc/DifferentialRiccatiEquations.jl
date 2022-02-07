_zeros(::Type{Matrix{T}}, m::Int, n::Int=m) where {T} = zeros(T, m, n)
_zeros(::Type{<:SparseArrays.SparseMatrixCSC{T}}, m::Int, n::Int=m) where {T} = spzeros(T, m, n)
