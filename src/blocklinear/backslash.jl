# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(
    prob::BlockLinearProblem,
    ::Backslash;
)
    @unpack A, B = prob
    X = A \ B
    return X
end
