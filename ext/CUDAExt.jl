# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module CUDAExt

using CUDA
using LinearAlgebra: Diagonal, svd

import DifferentialRiccatiEquations: orthf

# There is no (pivoted) QR on the GPU. Use the SVD instead:
function orthf(L::CuMatrix)
    F = svd(L)
    Q = F.U
    X = Diagonal(F.S) * F.Vt
    Q, X
end

end
