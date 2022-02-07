using DifferentialRiccatiEquations
using MAT, UnPack, Test
using LinearAlgebra
using SparseArrays

const DREs = DifferentialRiccatiEquations

# Dense Setup
P = matread(joinpath(@__DIR__, "Rail371.mat"))
@unpack E, A, B, C, X0 = P
Ed = collect(E) # d=dense
tspan = (4500., 4400.) # backwards in time
prob = GDREProblem(Ed, A, B, C, X0, tspan)

# Low-Rank Setup With Dense D
X0s = LDLᵀ(Matrix(C'), Matrix(100.0I(size(C, 1))))
sprob1 = GDREProblem(E, A, B, C, X0s, tspan)

# Low-Rank Setup With Sparse D
X0ss = LDLᵀ(Matrix(C'), sparse(100.0I(size(C, 1))))
sprob2 = GDREProblem(E, A, B, C, X0ss, tspan)

Δt(nsteps::Int) = (tspan[2] - tspan[1]) ÷ nsteps

@testset "DifferentialRiccatiEquations.jl" begin
    @testset "LDLᵀ" begin include("LDLt.jl") end

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

    @testset "Low-Rank $alg" for alg in (Ros1(),)
        # Replicate K with dense solver:
        ref  = solve(prob,   alg; dt=Δt(5))
        @testset "Dense D" begin
            smoketest(sprob1, alg)
            #=
            sol1 = solve(sprob1, alg; dt=Δt(5))
            @test ref.K[end] ≈ sol1.K[end]
            =#
        end
        @testset "Sparse D" begin
            smoketest(sprob2, alg)
            #=
            sol2 = solve(sprob2, alg; dt=Δt(5))
            @test ref.K[end] ≈ sol2.K[end]
            =#
        end
    end
end
