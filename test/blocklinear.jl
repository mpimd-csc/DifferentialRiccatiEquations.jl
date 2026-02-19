# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using DifferentialRiccatiEquations
using DifferentialRiccatiEquations: LinearProblem, LowRankUpdate
using LinearAlgebra
using SparseArrays

@isdefined(SimpleBiCGStab) || include("simple-bicgstab.jl")

function test_linear(prob, alg; reltol)
    A = prob.A
    B = rhs(prob)
    B_bak = copy(B)
    X = solve(prob, alg)
    # Ensure that the right-hand side has not been modified:
    @test B == B_bak
    @test norm(A*X - B) / norm(B) <= reltol
end

n = 10
k = 2
Ar = I + sprandn(n, n, 5/n)
Ac = I + sprandn(ComplexF64, n, n, 5/n)
U = randn(n, 1)
V = randn(n, 1)
B = randn(n, k)
b = randn(n)

reltol = 1e-10

@testset "$desc" for (desc, A) in (
    ("Real A", Ar),
    ("Real A + U*V'", LowRankUpdate(Ar, -1, U, V')),
    ("Complex A", Ac),
    ("Complex A + U*V'", LowRankUpdate(Ac, -1, U, V')),
)
    prob = LinearProblem(A, b)
    test_linear(prob, SimpleBiCGStab(; reltol); reltol)

    prob = BlockLinearProblem(A, B)
    test_linear(prob, Backslash(); reltol)
    test_linear(prob, ColumnWise(SimpleBiCGStab(; reltol)); reltol)
end
