# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

_diagm(T, v) = T(diagm(Vector(v)))
_diagm(::Type{<:Symmetric{S, T}}, v) where {S, T} = Symmetric(_diagm(T, v))
_diagm(::Type{<:Matrix}, v) = diagm(v)
_diagm(::Type{<:SparseMatrixCSC}, v) = spdiagm(v)
