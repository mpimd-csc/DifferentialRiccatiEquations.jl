# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module Stuff

using SparseArrays, LinearAlgebra

export orth, restrict

restrict(A::AbstractMatrix, Q) = Q' * A * Q

orth(N::SparseMatrixCSC) = orth(Matrix(N))

function orth(N::Matrix{T}) where {T}
    if VERSION < v"1.7"
        QR = qr(N, Val(true)) # pivoted
    else
        QR = qr(N, ColumnNorm())
    end
    R = QR.R
    # TODO: Find reference! As of LAPACK 3.1.2 or so,
    # the diagonal of R is sorted with decreasing absolute value,
    # and R is diagonal dominant. Therefore, it may be used to discover the rank.
    # Note that column permutations don't matter for span(N) == span(Q).
    ε = size(N, 1) * eps()
    r = 0
    for outer r in 1:size(R, 1)
        abs(R[r,r]) > ε && continue
        r -= 1
        break
    end
    Q = zeros(T, size(N, 1), r)
    for i in 1:r
        Q[i,i] = 1
    end
    lmul!(QR.Q, Q)
    return Q
end

end
