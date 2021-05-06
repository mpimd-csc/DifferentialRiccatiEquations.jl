using DifferentialRiccatiEquations
using Test

@testset "DifferentialRiccatiEquations.jl" begin
    @testset "$alg" for alg in (Ros1(), Ros2())
        include("rail.jl")
    end
end
