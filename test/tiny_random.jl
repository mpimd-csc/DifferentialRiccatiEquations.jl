# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using DifferentialRiccatiEquations
using LinearAlgebra, SparseArrays, UnPack

using DifferentialRiccatiEquations.Stuff: delta
using DifferentialRiccatiEquations.Shifts: Cyclic, Heuristic

n = 50
g = 4

function test_ale(E, A, g=g)
    n = size(E, 1)
    G = rand(n, g)
    S = -Matrix{Float64}(I(g))
    C = -2lowrank(G, S)

    prob = GALEProblem(E, A, C)
    res0 = norm(C)

    X_adi = solve(prob, ADI())
    X_ref = solve(prob, BartelsStewart())
    X_bad = solve(prob, Kronecker())
    X_gmres = solve(prob, GMRES(; maxiters=5, reltol=1e-8))
    X_fgmres = solve(prob, GMRES(;
        maxiters = 3,
        maxrestarts = 0,
        reltol = 1e-10, # default can not be reached
        preconditioner = ADI(;
            maxiters = 10,
            shifts = Cyclic(Heuristic(10, 10, 10)),
            compression_interval = 20, # only compress final solution
            warn_convergence = false,
        ),
    ))
    @test norm(residual(prob, X_ref)) / res0 < 1e-10
    @test norm(residual(prob, X_adi)) / res0 < 1e-10
    @test norm(residual(prob, X_bad)) / res0 < 0.02
    @test norm(residual(prob, X_gmres)) / res0 < 1e-8
    @test norm(residual(prob, X_fgmres)) / res0 < 1e-10

    @test delta(Matrix(X_adi), X_ref) < 1e-10
    @test delta(Matrix(X_gmres), X_ref) < 1e-8
    @test delta(Matrix(X_fgmres), X_ref) < 1e-10
    @test delta(X_bad, X_ref) < 0.02

    solver = init(prob, ADI())
    niter(solver) = length(solver.shifts)
    prev = 0
    @testset "ADI loop $i" for (i, _) in enumerate(solver)
        curr = niter(solver)
        @test prev + 1 <= curr <= prev + 2
        prev = curr
    end
    solver.last_compression > 0 && compress!(solver)
    @test solver.X == X_adi
end

@testset "Symmetric E" begin
    E = sprand(n, n, 1/n)
    E = E + E' + n*I
    @assert isposdef(E)

    @testset "Symmetric A" begin
        A = sprand(n, n, 1/n)
        A = A + A' - n*I
        @assert isposdef(-A)

        test_ale(E, A)
        test_ale(Symmetric(E), Symmetric(A))
        test_ale(Symmetric(E), A)
        test_ale(E, Symmetric(A))
    end

    @testset "Non-Symmetric A" begin
        A = sprand(n, n, 1/n)
        A = A - n*I
        @assert all(<(0), real.(eigvals(Matrix(A))))

        test_ale(E, A)
        test_ale(Symmetric(E), A)
    end
end

@testset "Non-Symmetric E" begin
    E = sprand(n, n, 1/n)
    E = E + n*I
    @assert all(>(0), real.(eigvals(Matrix(E))))

    @testset "Symmetric A" begin
        A = sprand(n, n, 1/n)
        A = A + A' - n*I
        @assert isposdef(-A)

        test_ale(E, A)
        test_ale(E, Symmetric(A))
    end

    @testset "Non-Symmetric A" begin
        A = sprand(n, n, 1/n)
        A = A - n*I
        @assert all(<(0), real.(eigvals(Matrix(A))))

        test_ale(E, A)
    end
end
