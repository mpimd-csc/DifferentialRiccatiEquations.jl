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

# Low-Rank Setup
q = size(C, 1)
L = E \ C'
D = Matrix(0.01I(q))
X0s = lowrank(L, D)
probs = GDREProblem(E, A, B, C, X0s, tspan)

# Dense Setup
X0 = Matrix(X0s)
prob = GDREProblem(E, A, B, C, X0, tspan)

# Verify Low-Rank Setup
@test E * X0 * E' ≈ C' * C / 100

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
    smoketest(probs, alg)
    sol = solve(probs, alg; dt=Δt(5))
    @test norm(ref.K[end] - sol.K[end]) < ε
end

@testset "Low-Rank Ros2()" begin
    alg = Ros2()
    # Replicate K with dense solver:
    ref = solve(prob, alg; dt=Δt(5))
    ε = norm(ref.K[end]) * size(E, 1) * eps() * 100
    smoketest(probs, alg)
    sol = solve(probs, alg; dt=Δt(5))
    @test norm(ref.K[end] - sol.K[end]) < ε
end

using DifferentialRiccatiEquations.Shifts

@testset "Newton-ADI" begin
    G = lowrank(B, I)
    Q = lowrank(C', I)
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
