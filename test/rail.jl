# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using DifferentialRiccatiEquations
using LinearAlgebra, SparseArrays
using UnPack
using MORWiki: SteelProfile, assemble

const DREs = DifferentialRiccatiEquations

# Ensure headless downloads succeed:
ENV["DATADEPS_ALWAYS_ACCEPT"] = 1

# Load system
@unpack E, A, B, C = assemble(SteelProfile(371))
B = Matrix(B)
C = Matrix(C)
tspan = (4500., 4400.) # backwards in time

# Low-Rank Setup With Dense D
q = size(C, 1)
L = E \ C'
D = Matrix(0.01I(q))
X0s = LDLᵀ(L, D)
sprob1 = GDREProblem(E, A, B, C, X0s, tspan)

# Low-Rank Setup With Sparse D
Ds = sparse(0.01I(q))
X0ss = LDLᵀ(L, Ds)
sprob2 = GDREProblem(E, A, B, C, X0ss, tspan)

# Dense Setup
X0 = Matrix(X0s)
prob = GDREProblem(E, A, B, C, X0, tspan)

# Verify Low-Rank Setup
@test E * X0 * E' ≈ C' * C / 100
@test Matrix(X0ss) ≈ X0

Δt(nsteps::Int) = (tspan[2] - tspan[1]) ÷ nsteps

function smoketest(prob, alg)
    sol = solve(prob, alg; dt=Δt(1))
    @test sol isa DREs.DRESolution
    @test length(sol.X) == 2 # only store first and last state by default
    @test first(sol.X) === prob.X0 # do not copy

    sol = solve(prob, alg; dt=Δt(2), save_state=true)
    @test sol isa DREs.DRESolution
    @test length(sol.t) == length(sol.X) == length(sol.K) == 3
    @test issorted(sol.t) == issorted(tspan) # do not alter direction of time
end

@testset "Dense $alg" for alg in (Ros1(), Ros2(), Ros3(), Ros4())
    smoketest(prob, alg)
end

@testset "Low-Rank Ros1()" begin
    alg = Ros1()
    # Replicate K with dense solver:
    ref = solve(prob, alg; dt=Δt(5))
    ε = norm(ref.K[end]) * size(E, 1) * eps() * 100
    @testset "Dense D" begin
        smoketest(sprob1, alg)
        sol1 = solve(sprob1, alg; dt=Δt(5))
        @test norm(ref.K[end] - sol1.K[end]) < ε
    end
    @testset "Sparse D" begin
        smoketest(sprob2, alg)
        sol2 = solve(sprob2, alg; dt=Δt(5))
        @test norm(ref.K[end] - sol2.K[end]) < ε
    end
end

@testset "Low-Rank Ros2()" begin
    alg = Ros2()
    # Replicate K with dense solver:
    ref = solve(prob, alg; dt=Δt(5))
    ε = norm(ref.K[end]) * size(E, 1) * eps() * 100
    @testset "Dense D" begin
        smoketest(sprob1, alg)
        sol1 = solve(sprob1, alg; dt=Δt(5))
        @test norm(ref.K[end] - sol1.K[end]) < ε
    end
    @testset "Sparse D" begin
        smoketest(sprob2, alg)
        sol2 = solve(sprob2, alg; dt=Δt(5))
        @test norm(ref.K[end] - sol2.K[end]) < ε
    end
end

using DifferentialRiccatiEquations.Shifts

@testset "Newton-ADI" begin
    G = LDLᵀ(B, I)
    Q = LDLᵀ(C', I)
    are = GAREProblem(E, A, G, Q)
    reltol = 1e-10
    @testset "$(adi_kwargs.shifts)" for adi_kwargs in [
        (shifts = Projection(2),), # leads to some complex shifts
        (shifts = Cyclic(Heuristic(10, 20, 20)), maxiters = 200),
    ]
        adi = ADI(; ignore_initial_guess=true, adi_kwargs...)
        newton = Newton(adi; maxiters=10, reltol)
        X = solve(are, newton)
        @test norm(residual(are, X)) < reltol * norm(Q)
    end
end
