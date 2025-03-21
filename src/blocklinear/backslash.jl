# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

struct BackslashSolver
    F
    B
end

function CommonSolve.init(
    prob::BlockLinearProblem,
    alg::Backslash;
)
    @unpack A, B = prob
    F = alg.factorize(A)
    BackslashSolver(F, B)
end

function CommonSolve.solve!(solver::BackslashSolver)
    @unpack F, B = solver
    X = F \ B
    return X
end
