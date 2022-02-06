using DifferentialRiccatiEquations
using MAT, UnPack, Test
using LinearAlgebra

const DREs = DifferentialRiccatiEquations

# Dense Setup
P = matread(joinpath(@__DIR__, "Rail371.mat"))
@unpack E, A, B, C, X0 = P
Ed = collect(E) # d=dense
tspan = (4500., 0.) # backwards in time
prob = GDREProblem(Ed, A, B, C, X0, tspan)

# Low-Rank Setup
X0s = LDLᵀ(Matrix(C'), Matrix(100.0I(size(C, 1))))
sprob = GDREProblem(E, A, B, C, X0s, tspan)

@testset "DifferentialRiccatiEquations.jl" begin
    @testset "LDLᵀ" begin include("LDLt.jl") end

    function smoketest(prob, alg)
        sol = solve(prob, alg; dt=-4500)
        @test sol isa DREs.DRESolution
        @test length(sol.X) == 2 # only store first and last state by default
        @test first(sol.X) === prob.X0 # do not copy

        sol = solve(prob, alg; dt=-2250, save_state=true)
        @test sol isa DREs.DRESolution
        @test length(sol.t) == length(sol.X) == length(sol.K) == 3
        @test issorted(sol.t) == issorted(tspan) # do not alter direction of time
    end

    @testset "Dense $alg" for alg in (Ros1(), Ros2(), Ros3(), Ros4())
        smoketest(prob, alg)
    end

    @testset "Low-Rank $alg" for alg in (Ros1(),)
        smoketest(sprob, alg)

        #= DONT RUN THIS!
        # Without column compression the matrices get HUGE, 371x4926 and more.
        # I need to implement compression before I can test accuracy.

        # Replicate K with dense solver:
        ref = solve(prob, alg; dt=-500)
        sol = solve(sprob, alg; dt=-500)
        @test ref.K[end] ≈ sol.K[end]
        =#
    end
end
