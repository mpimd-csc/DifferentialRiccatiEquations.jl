# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

_diagm(::Type{<:Matrix}, v) = diagm(v)
_diagm(::Type{<:SparseMatrixCSC}, v) = spdiagm(v)
