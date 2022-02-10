using Test, DifferentialRiccatiEquations
using DifferentialRiccatiEquations: compress
using LinearAlgebra

n = 10
k = 3

@testset "Conversions" begin
    L = randn(n, k)
    D = rand(k, k)
    X = LDLᵀ(L, D)
    M = Matrix(X)

    @test M isa Matrix{Float64}
    @test size(M) == (n, n)
    @test M ≈ L*D*L'
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

    for d in 2:k
        X.D[d,d] = 0
    end
    X = compress(X)
    @test rank(X) == 1
end

@testset "Rank 0" begin
    X = LDLᵀ(randn(n, 1), zeros(1, 1))
    @test rank(X) == 1 # not the actual rank
    X = compress(X)
    @test rank(X) == 0
    @test Matrix(X) == zeros(n, n)

    X = sample(n, 0)
    @test rank(X) == 0
    @test Matrix(X) == zeros(n, n)

    X = LDLᵀ{Matrix{Bool},Matrix{Bool}}(n, 0)
    @test rank(X) == 0
end

@testset "Arithmetic" begin
    X = sample(n, k)
    @test rank(X) == k
    @test rank(X+X) == k
    #@test rank(X-X) == 0 # flaky
end
