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
        sol = solve(prob, alg; dt=-1500)
        @test sol isa DREs.DRESolution
    end
end
