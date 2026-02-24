# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using Adapt: adapt
using CUDA
using DifferentialRiccatiEquations
using KernelAbstractions
using LinearAlgebra, SparseArrays

function test_residual_zero(prob, C)
    res = residual(prob, zero(C))
    @test typeof(res) == typeof(C)
    @test res == C
    @test res !== C # must be safe to modify in-place

    backend = get_backend(prob.A)
    mat0 = KernelAbstractions.zeros(backend, eltype(res), n, n)
    res0 = norm(C)
    @test norm(res) ≈ res0
    @test norm(residual(prob, mat0)) ≈ res0
end

function test_residual_rand(prob, X)
    resX = residual(prob, X)
    resM = residual(prob, AbstractMatrix(X))
    @test typeof(resX) <: DifferentialRiccatiEquations.LDLᵀ
    @test eltype(resX) == eltype(X)
    @test eltype(resX.Ls) == eltype(X.Ls)
    @test norm(resX) ≈ norm(resM)

    eltype(X.Ds) <: UniformScaling && return
    @test eltype(resX.Ds) == eltype(X.Ds)
    @test typeof(resX) == typeof(X)
end

assemble_factor(desc::Symbol, type::Type, dim::Int) = assemble_factor(Val(desc), type, dim)
assemble_factor(::Val{:definite}, _, _) = I
assemble_factor(::Val{:scaled}, T, _) = 2 * one(T) * I
function assemble_factor(::Val{:indefinite}, T, s)
    S = one(T) * I(s)
    S[:, end:-1:begin]
end

T = Float64 # eltype
n = 20 # size of system matrices
c = 4 # rank of constant term
q = 3 # rank of quadratic term
z = 2 # rank of random solution
inner_factors = (:definite, :scaled, :indefinite)
backends = Any[ # prefix, outer-factor backend, inner-factor backend
    ("", CPU(), CPU()),
]
CUDA.functional() && append!(backends, [
    ("CUDA: ", CUDABackend(), CUDABackend()),
    ("CUDA+CPU: ", CUDABackend(), CPU()),
])

# Assemble system matrices:
Eh = sprand(T, n, n, 1/n)
Ah = sprand(T, n, n, 1/n)

# Assemble random solution:
Zh = rand(n, z)
Yh = assemble_factor(:indefinite, T, z)

@testset "$(prefix)GALEProblem" for (prefix, TO, to) in backends
    E = adapt(TO, Eh)
    A = adapt(TO, Ah)
    Z = adapt(TO, Zh)
    Y = adapt(to, Yh)
    @testset "$desc C" for desc in inner_factors
        G = adapt(TO, randn(n, c))
        S = adapt(to, assemble_factor(desc, T, c))
        C = lowrank(G, S)
        prob = GALEProblem(E, A, C)
        @testset "zero X" test_residual_zero(prob, C)
        @testset "definite X" test_residual_rand(prob, 2lowrank(Z))
        @testset "indefinite X" test_residual_rand(prob, 2lowrank(Z, Y))
    end
end;

@testset "$(prefix)GAREProblem" for (prefix, TO, to) in backends
    E = adapt(TO, Eh)
    A = adapt(TO, Ah)
    Z = adapt(TO, Zh)
    Y = adapt(to, Yh)
    @testset "$desc C" for desc in inner_factors
        C_outer = adapt(TO, rand(n, c))
        C_inner = adapt(to, assemble_factor(desc, T, c))
        C_inner = adapt(to, C_inner)
        C = lowrank(C_outer, C_inner)
        @testset "$desq Q" for desq in inner_factors
            Q_outer = adapt(TO, rand(n, q))
            Q_inner = adapt(to, assemble_factor(desq, T, q))
            Q = lowrank(Q_outer, Q_inner)
            prob = GAREProblem(E, A, Q, C)
            @testset "zero X" test_residual_zero(prob, C)
            @testset "definite X" test_residual_rand(prob, 2lowrank(Z))
            @testset "indefinite X" test_residual_rand(prob, 2lowrank(Z, Y))
        end
    end
end;
