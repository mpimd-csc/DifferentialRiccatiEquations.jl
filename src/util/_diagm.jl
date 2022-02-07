_diagm(::Type{<:Matrix}, v) = diagm(v)
_diagm(::Type{<:SparseMatrixCSC}, v) = spdiagm(v)
