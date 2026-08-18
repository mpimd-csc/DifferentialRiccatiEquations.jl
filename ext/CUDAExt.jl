# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module CUDAExt

using CUDA, CUDA.CUSPARSE
using LinearAlgebra: Adjoint, Diagonal, UniformScaling, checksquare, diagind, svd

import DifferentialRiccatiEquations: assemble_shifted_system, orthf
import LinearAlgebra

const RelevantGPUSparseArray = Union{
    CuSparseMatrixCOO{T},
    CuSparseMatrixCSC{T},
    CuSparseMatrixCSR{T},
} where {T}

# This should not be needed be the default for `::Adjoint`...
function assemble_shifted_system(
    A::Adjoint{T, <: RelevantGPUSparseArray{T}},
    E::Adjoint{T, <: RelevantGPUSparseArray{T}},
    μ,
) where {T}
    (A' + E' * conj(μ))'
end

# There is no (pivoted) QR on the GPU. Use the SVD instead:
function orthf(L::CuMatrix)
    F = svd(L)
    Q = F.U
    X = Diagonal(F.S) * F.Vt
    Q, X
end

end
