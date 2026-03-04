# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test, DifferentialRiccatiEquations
using Adapt: adapt
using CUDA
using KernelAbstractions: get_backend
using LinearAlgebra

function sample(n, k)
    for _ in 1:10
        L = randn(n, k)
        rank(L) == k && return L
    end
    error("Failed to generate full-rank matrix")
end

function test_lowrank_essentials(X, Uref, Sref, Vref)
    @assert all(issymmetric, X.Ds)

    @test eltype(X) == Float64
    @test size(X) == (n, n)
    @test rank(X) == k

    @testset "Conversion to $T" for T in (Matrix, Matrix{Float64}, Matrix{Float32})
        Srefx = Sref isa UniformScaling ? Sref : adapt(get_backend(Uref), Sref)

        M = T(X)
        @test M isa T
        @test size(M) == (n, n)
        @test M ≈ convert(Matrix, Uref * Srefx * Vref')
        @test M == convert(T, X)
    end

    function test_destructure(X)
        # TODO: Adjust to `alpha, Z1, Y, Z2 = X` once implemented
        alpha, Z1, Y = X
        @test alpha === 1.0
        @test Z1 === Uref
        @test Y === Sref
    end

    # Destructure through iteration:
    test_destructure(X)

    # Repeated iteration does not alter the objects:
    test_destructure(X)
end

function test_lowrank_arithmetic(X)
    isdefinite = eltype(X.Ds) <: UniformScaling
    @test !iszero(X)

    T = typeof(X)
    @test 2X isa T
    @test -X isa T broken=isdefinite
    @test X + X isa T
    @test X - X isa T broken=isdefinite

    Y = 2X
    @test Y isa T
    @test Y.alphas == 2 * X.alphas
    @test Y.Ls === X.Ls
    @test Y.Ds === X.Ds
    @test norm(Y) ≈ 2norm(X)

    M = Matrix(X)
    @test rank(X) == rank(M) == k
    @test norm(X) ≈ norm(M)
    @test Matrix(2X + 3X) ≈ 5M
    @test norm(Matrix(X - X)) / eps() < 10n broken=isdefinite

    Z = zero(X)
    @test Z isa T
    @test rank(Z) == 0
    @test iszero(Z)
    @test Matrix(Z) == zeros(n, n)
    @test X + Z == X
    @test Z + X == X
end

function test_lowrank_compression(X)
    Y = compress!(X + X)
    @test Y isa typeof(X)
    @test rank(X) == k
    @test rank(Y) == k
    @test Matrix(Y) ≈ 2Matrix(X)

    eltype(X.Ds) <: UniformScaling && return
    X = deepcopy(X)
    S = only(X.Ds)
    fill!(S, 0)
    CUDA.@allowscalar S[1, 1] = 13
    @test rank(X) == k # storage size != numerical rank
    @test rank(compress!(X)) == 1
end

n = 10
k = 2

U = sample(n, k)
V = sample(n, k) # TODO
S = sample(k, k)
S += S'
@assert issymmetric(S)

tests = Any[
    ("Symmetric definite U * U'", lowrank(U), U, I, U),
    ("Symmetric indefinite U * (S::Matrix) * U'", lowrank(U, S), U, S, U),
    ("Symmetric indefinite U * (S::Symmetric) * U'", lowrank(U, Symmetric(S)), U, Symmetric(S), U),
    # TODO: ("Nonsymmetric U * V'", lowrank(U, I, V), U, I, V),
    # TODO: ("Nonsymmetric U * S * V'", lowrank(U, S, V), U, S, V),
]

if CUDA.functional()
    to = CUDABackend()
    Ud = adapt(to, U)
    Sd = adapt(to, S)
    append!(tests, [
        ("CUDA: Symmetric definite U * U'", lowrank(Ud), Ud, I, Ud),
        ("CUDA: Symmetric indefinite U * (S::Matrix) * U'", lowrank(Ud, S), Ud, S, Ud),
        ("CUDA: Symmetric indefinite U * (S::CuMatrix) * U'", lowrank(Ud, Sd), Ud, Sd, Ud),
        # FIXME: ("CUDA: Symmetric indefinite U * (S::Symmetric) * U'", lowrank(Ud, Symmetric(Sd)), Ud, Symmetric(Sd), Ud),
    ])

    @testset "Adapt" begin
        X1 = adapt(to, lowrank(U))
        X2 = lowrank(Ud)
        @test typeof(X1) == typeof(X2)
        @test X1 == X2

        X1 = adapt(to, lowrank(U, S))
        X2 = lowrank(Ud, Sd)
        @test typeof(X1) == typeof(X2)
        @test X1 == X2

        X1 = adapt(to, lowrank(U, Symmetric(S)))
        X2 = lowrank(Ud, Symmetric(Sd))
        @test typeof(X1) == typeof(X2)
        @test X1 == X2
    end
end;

@testset "$desc" for (desc, X, Uref, Sref, Vref) in tests
    @testset "Essentials" test_lowrank_essentials(X, Uref, Sref, Vref)
    @testset "Arithmetic" test_lowrank_arithmetic(X)
    if !(eltype(X.Ds) <: UniformScaling)
        @testset "Compression" test_lowrank_compression(X)
    end
end;
