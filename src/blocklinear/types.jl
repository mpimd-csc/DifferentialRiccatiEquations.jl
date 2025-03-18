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

abstract type BlockLinearSolver end
struct Backslash <: BlockLinearSolver
    factorize
    Backslash(f=identity) = new(f)
end
struct ShermanMorrisonWoodbury <: BlockLinearSolver
    ALG::BlockLinearSolver
    alg::BlockLinearSolver
    ShermanMorrisonWoodbury(ALG=Backslash(), alg=Backslash()) = new(ALG, alg)
end
