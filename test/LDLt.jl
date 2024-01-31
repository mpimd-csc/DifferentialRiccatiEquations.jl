# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test, DifferentialRiccatiEquations
using LinearAlgebra

n = 10
k = 2

@testset "Conversions" begin
    L = randn(n, k)
    D = rand(k, k)
    X = LDLᵀ(L, D)
    M = Matrix(X)

    @test M isa Matrix{Float64}
    @test size(M) == (n, n)
    @test M ≈ L*D*L'
    @test norm(M) ≈ norm(X)
end

function sample(n, k)
    local D, L
    while true
        L = randn(n, k)
        rank(L) == k && break
    end
    λ = rand(k) .+ 0.1
    D = diagm(λ)
    return LDLᵀ(L, D)
end

@testset "Rank k" begin
    X = sample(n, k)
    M = Matrix(X)
    @test M isa Matrix{Float64}
    @test rank(X) == rank(M) == k

    _L = only(X.Ls)
    _D = only(X.Ds)
    L, D = X
    @test L === _L
    @test D === _D

    for d in 2:k
        D[d,d] = 0
    end
    compress!(X)
    @test rank(X) == 1
end

@testset "Rank 0" begin
    X = LDLᵀ(randn(n, 1), zeros(1, 1))
    @test rank(X) == 1 # not the actual rank
    compress!(X)
    @test rank(X) == 0
    @test Matrix(X) == zeros(n, n)

    X = sample(n, 0)
    @test rank(X) == 0
    @test Matrix(X) == zeros(n, n)

    Z = zero(X)
    @test typeof(Z) == typeof(X)
    @test rank(Z) == 0
end

@testset "Compression" begin
    # TODO: Once compression is configurable, this must be adjusted.
    @assert 2k < 0.5n # U+U does not trigger compression

    U = sample(n, k)
    V = Matrix(U)
    W = U + U
    @test rank(V) == k
    @test rank(W) == 2k
    @test Matrix(W) ≈ 2V

    @testset "Implicit Compression" begin
        # Implicit compression upon iteration:
        W = U + U
        @test rank(W) == 2k
        L, D = W
        @test rank(W) == k
        @test size(L, 1) == n
        @test size(L, 2) == size(D, 1) == size(D, 2) == k
        @test Matrix(W) ≈ L*D*L' ≈ 2V

        # Repeated iteration does not alter the components:
        L1, D1 = W
        @test L1 === L
        @test D1 === D
    end

    @testset "Skipped Compression" begin
        # Don't compress singleton components:
        W = U + U
        concatenate!(W)
        @test rank(W) == 2k
        L, D = W
        @test size(L, 1) == n
        @test size(L, 2) == size(D, 1) == size(D, 2) == 2k
        @test Matrix(W) ≈ L*D*L' ≈ 2V
    end

    desc = ("w/ ", "w/o")
    concat = (true, false)
    @testset "Explicit Compression $d Concatenation" for (d, cc) in zip(desc, concat)
        # Explicit compression reduces rank:
        W = U + U
        cc && concatenate!(W)
        @test rank(W) == 2k
        compress!(W)
        @test rank(W) == k
        L, D = W
        @test size(L, 1) == n
        @test size(L, 2) == size(D, 1) == size(D, 2) == k
        @test Matrix(W) ≈ L*D*L' ≈ 2V
    end
end

@testset "Arithmetic" begin
    # TODO: Once compression is configurable, this must be adjusted.
    @assert 3k > 0.5n # X+X+X does trigger compression

    X = sample(n, k)
    @test rank(X) == k

    @test rank(compress!(X+X)) == k
    @test rank(X+X+X) == k
    #@test rank(X-X) == 0 # flaky
end
