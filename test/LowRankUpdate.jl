# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using LinearAlgebra
using DifferentialRiccatiEquations: LowRankUpdate, lr_update
using SparseArrays

n = 10
k = 3

@testset "Dense A" begin
    A = rand(n, n)
    U = rand(n, k)
    V = rand(k, n)

    AUV = lr_update(A, 1, U, V)
    @test AUV isa Matrix
end

function test_lr_update(A, U, V)
    AUV = lr_update(A, -1, U, V) # technically a downdate
    @test AUV isa LowRankUpdate

    # Decomposition:
    _A, _α, _U, _V = AUV
    @test _A === A
    @test _U === U
    @test _V === V
    @test _α === -1

    # Vector solve:
    M = A - U*V
    B = rand(n)
    X = AUV \ B
    @test M * X ≈ B

    # Matrix solve:
    B1 = rand(n, 1)
    X1 = AUV \ B1
    @test M * X1 ≈ B1

    # Addition:
    E = sprand(n, n, 0.2)
    EUV = AUV + E
    @test typeof(EUV) == typeof(AUV)
    _E, _α, _U, _V = EUV
    @test _E ≈ A + E
    @test _U === U
    @test _V === V
    @test _α === -1
end

@testset "Sparse A" begin
    A = spdiagm(1.0:n) # invertible
    A[2,1] = 1 # not diagonal
    A[3,2] = 2 # not triangular or symmetric

    desc = ["Dense", "Sparse"]
    Us = [rand(n, k), sprand(n, k, 0.5)]
    Vs = [rand(k, n), sprand(k, n, 0.5)]
    tests = Iterators.product(
        zip(desc, Us),
        zip(desc, Vs),
    )

    @testset "$descU U, $descV V" for ((descU, U), (descV, V)) in tests
        test_lr_update(A, U, V)
    end
end
