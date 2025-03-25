# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using DifferentialRiccatiEquations
using LinearAlgebra, SparseArrays

n = 20
E = sprand(n, n, 1/n)
A = sprand(n, n, 1/n)
C = LDLᵀ(rand(n, 4), rand(4, 4)) # constant term
Q = LDLᵀ(rand(n, 3), rand(3, 3)) # quadratic term

res0 = norm(C)

@testset "$desc" for (desc, prob) in (
    ("GALEProblem", GALEProblem(E, A, C)),
    ("GAREProblem", GAREProblem(E, A, Q, C)),
)
    @test residual(prob, zero(C)) == C
    @test residual(prob, zero(C)) !== C # must be safe to modify in-place
    @test norm(residual(prob, zero(C))) ≈ res0
    @test norm(residual(prob, zeros(n, n))) ≈ res0
end
