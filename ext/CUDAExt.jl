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

# Adapted from uniformscaling.jl:287 as of Julia 1.10.9
# TODO: Discuss this change upstream:
function LinearAlgebra.mul!(out::SubArray{T, 2, <: CuMatrix{T}}, a::Number, B::UniformScaling, α::Number, β::Number) where {T}
    checksquare(out)
    if iszero(β)  # zero contribution of the out matrix
        fill!(out, zero(T))
    elseif !isone(β)
        rmul!(out, β)
    end
    s = convert(T, a*B.λ*α)
    if !iszero(s)
        ids = diagind(out)
        out[ids] .+= s # allocates :-(
    end
    return out
end

end
