# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using LinearAlgebra: kron

function CommonSolve.solve(prob::GALEProblem, ::Kronecker)
    @unpack E, A, C = prob

    F = kron(E', A) + kron(A', E)
    b = -vec(Matrix(C))
    x = F \ b

    n = size(E, 1)
    X = reshape(x, n, n)
    return X
end