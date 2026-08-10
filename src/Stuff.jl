# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module Stuff

using SparseArrays, LinearAlgebra

export orth, restrict

# TODO: Is Q orthogonal for all uses? If so, optimize the UniformScaling method.
restrict(A, Q) = Q' * A * Q
restrict(A::UniformScaling, Q) = A * (Q' * Q)

orth(N::SparseMatrixCSC) = orth(Matrix(N))

function orth(N::AbstractMatrix)
    F = svd(N)
    ε = size(N, 1) * eps()
    ids = findall(s -> abs(s) > ε, F.S)
    F.U[:, ids]
end

# Only for testing:
delta(a, b) = norm(a - b) / max(norm(a), norm(b))

end
