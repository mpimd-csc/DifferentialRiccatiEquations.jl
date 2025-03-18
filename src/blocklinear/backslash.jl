# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(
    prob::BlockLinearProblem,
    alg::Backslash;
)
    @unpack A, B = prob
    F = alg.factorize(A)
    X = F \ B
    return X
end
