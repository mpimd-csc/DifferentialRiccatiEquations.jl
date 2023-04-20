# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using Test
using DifferentialRiccatiEquations
using SparseArrays
using DifferentialRiccatiEquations: orth

@testset "DifferentialRiccatiEquations.jl" begin
    @testset "LDLᵀ" begin include("LDLt.jl") end
    @testset "LowRankUpdate" begin include("LowRankUpdate.jl") end
    @testset "orth" begin
        N = zeros(4, 1)
        Q = orth(N)
        @test size(Q) == (4, 0)
        Ns = sparse(N)
        Qs = orth(Ns)
        @test size(Qs) == (4, 0)
    end
    @testset "Oberwolfach Rail" begin include("rail.jl") end
end
