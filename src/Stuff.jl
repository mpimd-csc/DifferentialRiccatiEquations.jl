# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module Stuff

using SparseArrays, LinearAlgebra

export orth, restrict

restrict(A, Q) = Q' * A * Q

orth(N::SparseMatrixCSC) = orth(Matrix(N))

function orth(N::AbstractMatrix)
    F = svd(N)
    ε = size(N, 1) * eps()
    ids = findall(s -> abs(s) > ε, F.S)
    F.U[:, ids]
end

end
