using DifferentialRiccatiEquations
using MAT, UnPack, Test

const DREs = DifferentialRiccatiEquations

# Setup
P = matread(joinpath(@__DIR__, "Rail371.mat"))
@unpack E, A, B, C, X0 = P
Ed = collect(E) # d=dense
tspan = (4500., 0.) # backwards in time
prob = GDREProblem(Ed, A, B, C, X0, tspan)

@testset "DifferentialRiccatiEquations.jl" begin
    @testset "$alg" for alg in (Ros1(), Ros2(), Ros3(), Ros4())
        sol = solve(prob, alg; dt=-4500)
        @test sol isa DREs.DRESolution
        @test length(sol.X) == 2 # only store first and last state by default
        @test first(sol.X) === X0 # do not copy

        sol = solve(prob, alg; dt=-2250, save_state=true)
        @test sol isa DREs.DRESolution
        @test length(sol.t) == length(sol.X) == length(sol.K) == 3
        @test issorted(sol.t) == issorted(tspan) # do not alter direction of time
    end
end
