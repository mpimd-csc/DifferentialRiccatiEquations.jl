# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

struct ColumnWiseSolver
    A
    B
    cache
end

solution(colwise::ColumnWiseSolver) = colwise.X

function CommonSolve.init(
    prob::BlockLinearProblem,
    colwise::ColumnWise;
    kwargs...,
)
    (; A, B) = prob
    b = similar(B, size(B, 1))
    prob = LinearProblem(A, b)
    cache = init(prob, colwise.alg; kwargs...)
    ColumnWiseSolver(A, B, cache)
end

function CommonSolve.solve!(colwise::ColumnWiseSolver)
    (; A, B, cache) = colwise

    # Currently, the solution will become the outer low-rank factor of an ADI increment.
    # Hence, we must allocate a new solution for every solve,
    # which is a bit unfortunate in terms of reusability.
    # Not requiring to allocate a new solution object here (without imposing unnecessary allocations elsewhere) is a non-trivial task (i.e. TODO).
    X = similar(B, eltype(A))

    for (x, b) in zip(eachcol(X), eachcol(B))
        copyto!(rhs(cache), b)
        solve!(cache)
        copyto!(x, solution(cache))
    end

    return X
end
