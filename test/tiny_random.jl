# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using Adapt: adapt
using CUDA
using DifferentialRiccatiEquations
using KernelAbstractions: CPU, get_backend
using LinearAlgebra, SparseArrays, UnPack
using StableRNGs

using DifferentialRiccatiEquations.Stuff: delta
using DifferentialRiccatiEquations.Shifts: Cyclic, Heuristic

import DifferentialRiccatiEquations.Shifts: isstable

@isdefined(SimpleBiCGStab) || include("simple-bicgstab.jl")

n = 50
g = 4

# TODO: Test mixed-backend LDLᵀ

# TODO: Refactor to property-based tests:
# 1. Compute solution using ADI(; reltol=1e-4)
# 2. Compute normalized residual resnorm
# 3. Test resnorm <= 1e-4
# 4. Compute new solution using ADI(; reltol=resnorm/10)
# 5. Test new resnorm <= new reltol; note that new reltol is smaller than old reltol

LinearAlgebra.factorize(A::UniformScaling) = A

function test_ale(E, A, g=g)
    backend = get_backend(A)

    n = size(A, 1)
    G = rand(StableRNG(42), n, g)
    G = adapt(backend, G)
    S = -Matrix{Float64}(I(g))
    C = -2lowrank(G, S)

    prob = GALEProblem(E, A, C)
    res0 = norm(C)

    # Compute reference solution:
    prob_cpu = adapt(CPU(), prob)
    X_ref = solve(prob_cpu, BartelsStewart())

    # Test GPU-ready algorithms:
    X_adi2 = solve(prob, ADI(ColumnWise(SimpleBiCGStab())))
    @test norm(residual(prob, X_adi2)) / res0 < 1e-10
    @test delta(Matrix(X_adi2), X_ref) < 1e-10

    @testset "ADI progresses by 1 or 2 steps at a time" begin
        solver = init(prob, ADI(ColumnWise(SimpleBiCGStab())))
        niter(solver) = length(solver.shifts)
        prev = 0
        @testset "ADI loop $i" for (i, _) in enumerate(solver)
            curr = niter(solver)
            @test prev + 1 <= curr <= prev + 2
            prev = curr
        end
        solver.last_compression > 0 && compress!(solver)
        #@test solver.X == X_adi2
    end

    # Most solvers don't work on the GPU (yet). :-(
    backend isa CPU || return

    X_adi = solve(prob, ADI())
    X_adi3 = solve(prob, ADI(ColumnWise(SimpleBiCGStab()); inexact=true))
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
    @test norm(residual(prob, X_adi3)) / res0 < 1e-10
    @test norm(residual(prob, X_bad)) / res0 < 1e-10
    @test norm(residual(prob, X_gmres)) / res0 < 1e-8
    @test norm(residual(prob, X_fgmres)) / res0 < 1e-10

    @test delta(Matrix(X_adi), X_ref) < 1e-10
    @test delta(Matrix(X_adi3), X_ref) < 1e-10
    @test delta(Matrix(X_gmres), X_ref) < 1e-8
    @test delta(Matrix(X_fgmres), X_ref) < 1e-10
    @test delta(X_bad, X_ref) < 0.02
end

isstable(X::SparseMatrixCSC) = isstable(Matrix(X))
isstable(X::AbstractMatrix) = all(isstable, eigvals(X))

rng = StableRNG(123)

E = sprand(rng, n, n, 1/n)
E_sym = E + E' + n*I
E_gen = E + n*I
@assert isposdef(E_sym)
@assert isstable(-E_gen)
@assert !issymmetric(E_gen)

A = sprand(rng, n, n, 1/n)
A_sym = A + A' - n*I
A_gen = A - n*I
@assert isposdef(-A_sym)
@assert isstable(A_gen)
@assert !issymmetric(A_gen)

Es = Any[ # desc, E
    ("Symmetric E", E_sym),
    ("E::Symmetric", Symmetric(E_sym)),
    ("Non-symmetric E", E_gen),
    ("Identity E", I),
    ("E::UniformScaling", 2 * I),
]

As = Any[ # desc, A
    ("Symmetric A", A_sym),
    ("A::Symmetric", Symmetric(A_sym)),
    ("Non-symmetric A", A_gen),
]

tests = Any[] # desc, E, A
for (descE, E) in Es, (descA, A) in As
    push!(tests, ("$descE, $descA", E, A))
end
# TODO: push!(tests, ("...", E_gen', A_gen')) or do I want all combinations?
if CUDA.functional()
    to = CUDABackend()
    for (descE, E) in Es
        E isa UniformScaling && continue # TODO
        E isa Symmetric && continue # FIXME
        Ed = adapt(to, E)
        for (descA, A) in As
            A isa Symmetric && continue # FIXME
            Ad = adapt(to, A)
            push!(tests, ("CUDA: $descE, $descA", Ed, Ad))
        end
    end
end

@testset "ADI w/o assembling the solution" begin
    E, A = E_sym, A_sym
    n = size(A, 1)
    rng = StableRNG(21)
    G = randn(rng, n, g)
    S = randn(rng, g, g)
    S += S'
    C = lowrank(G, S)

    prob = GALEProblem(E, A, C)
    reltol = 1e-4
    alg = ADI(; reltol)

    # Ensure input is unchanged:
    initial_guess = zero(C)
    X = solve(prob, alg; initial_guess, accumulate_solution=false)
    @test X === initial_guess
    @test iszero(X)

    # Ensure ADI still solves equation:
    cache = init(prob, alg; accumulate_solution=false)
    X = zero(C)
    for _ in cache
        X += cache.increment
    end
    @test norm(residual(prob, X)) / norm(C) <= reltol
    @test iszero(solve!(cache))
end

@testset "$desc" for (desc, E, A) in tests
    test_ale(E, A)
end;
