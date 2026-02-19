# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using DifferentialRiccatiEquations
using DifferentialRiccatiEquations: LinearProblem, LowRankUpdate
using LinearAlgebra
using LinearSolve: KrylovJL_BICGSTAB
using SparseArrays
using StableRNGs

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

rng = StableRNG(42)
n = 100
k = 4
r = 2
Ar = 2I + sprandn(rng, n, n, 5/n)
Ac = 2I + sprandn(rng, ComplexF64, n, n, 5/n)
U = randn(rng, n, r)
V = randn(rng, n, r)
B = randn(rng, n, k)
b = randn(rng, n)

reltol = 1e-10
maxiters = 2n
simple_bicgstab = SimpleBiCGStab(; maxiters, reltol, rng=StableRNG(42))
krylov_bicgstab = WithKwargs(KrylovJL_BICGSTAB(); maxiters, reltol)

@testset "$desc" for (desc, A) in (
    ("Real A", Ar),
    ("Real A + U*V'", LowRankUpdate(Ar, -1, U, V')),
    ("Complex A", Ac),
    ("Complex A + U*V'", LowRankUpdate(Ac, -1, U, V')),
)
    prob = LinearProblem(A, b)
    test_linear(prob, simple_bicgstab; reltol)
    test_linear(prob, krylov_bicgstab; reltol = 20reltol)

    prob = BlockLinearProblem(A, B)
    test_linear(prob, Backslash(); reltol)
    test_linear(prob, ColumnWise(simple_bicgstab); reltol)
    test_linear(prob, ColumnWise(krylov_bicgstab); reltol = 20reltol)
end
