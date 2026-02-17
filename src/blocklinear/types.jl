# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
Linear system with multiple right-hand sides

    AX = B

having fields `A` and `B`.
"""
struct BlockLinearProblem
    A
    B
end

# Avoid dependency on SciMLBase:
"""
Linear system with a single right-hand side

    Ax = b

having fields `A` and `b`.
"""
struct LinearProblem
    A
    b
end

rhs(prob::LinearProblem) = prob.b

"""
Default interface:

```julia
CommonSolve.init(prob::BlockLinearProblem, alg::BlockLinearSolver) -> solver
CommonSolve.solve!(solver)
rhs(solver) = solver.B # must be in-place modifiable
```

If `alg::BlockLinearSolver` does only support `CommonSolve.solve` but not `init` and `solve!`,
there is a fallback implementation in the form of `DefaultBlockLinearSolver`.

The object returned by `rhs` must be in-place modifiable.
More specifically, it must support `mul!(rhs(solver), A, X)` for some `A` and `X`.
"""
abstract type BlockLinearSolver end
struct Backslash <: BlockLinearSolver
    factorize
    Backslash(f=factorize_unless_factorized) = new(f)
end

"""
    ColumnWise(alg)::BlockLinearSolver

Solve a block-linear system column by column.
The inner algorithm `alg` must permit the following usage:

```julia
import DifferentialRiccatiEquations as DRE

cache = CommonSolve.init(prob::DRE.LinearProblem, alg; abstol)
CommonSolve.solve!(cache)

b = DRE.rhs(cache) # must be in-place modifiable
x = DRE.solution(cache) # safe to alias internal state
```

In particular, we following must be allowed:

```julia
copyto!(DRE.rhs(cache), _)
copyto!(_, DRE.solution(cache))
```

Furthermore, `init(_, alg)` will only be called once overall,
but `solve!(cache)` will be called multiple times while updating `rhs(cache)` in between.

See also: [`BlockLinearProblem`](@ref), [`LinearProblem`](@ref), [`rhs`](@ref), [`solution`](@ref)
"""
struct ColumnWise <: BlockLinearSolver
    alg
end

struct ShermanMorrisonWoodbury <: BlockLinearSolver
    ALG::BlockLinearSolver
    alg::BlockLinearSolver
    ShermanMorrisonWoodbury(ALG=Backslash(), alg=Backslash()) = new(ALG, alg)
end

factorize_unless_factorized(X) = factorize(X)
factorize_unless_factorized(X::Factorization) = X

### Fallback

struct DefaultBlockLinearSolver
    B
    prob
    alg
end

function CommonSolve.init(prob::BlockLinearProblem, alg::BlockLinearSolver)
    DefaultBlockLinearSolver(copy(prob.B), prob, alg)
end

function CommonSolve.solve!(solver::DefaultBlockLinearSolver)
    @unpack B, prob, alg = solver
    prob = BlockLinearProblem(prob.A, B)
    CommonSolve.solve(prob, alg)
end

rhs(s) = s.B
