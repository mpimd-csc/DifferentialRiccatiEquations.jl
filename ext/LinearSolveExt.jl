# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module LinearSolveExt

using DifferentialRiccatiEquations: init
using LinearAlgebra: mul!

import CommonSolve
import DifferentialRiccatiEquations as DRE
import LinearSolve as LS

# Implement matrix-free interface expected by `LS.LinearProblem`:
(A::DRE.LowRankUpdate)(Ax, x, _, _) = mul!(Ax, A, x)

function CommonSolve.init(
    prob::DRE.LinearProblem,
    alg::LS.SciMLLinearSolveAlgorithm;
    kwargs...,
)
    (; A, b) = prob
    T = eltype(A)
    b = convert(AbstractArray{T}, b)
    init(LS.LinearProblem(A, b), alg; kwargs...)
end

DRE.rhs(cache::LS.LinearCache) = cache.b
DRE.solution(cache::LS.LinearCache) = cache.u

end
