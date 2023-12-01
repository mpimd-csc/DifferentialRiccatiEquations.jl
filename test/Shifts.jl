# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using SparseArrays
using Test

using DifferentialRiccatiEquations: Shifts
using .Shifts
using .Shifts: take!, init

penzl(p) = [-1 p; -p -1]

n = 3
E = sparse(1.0I(n))
A = spzeros(n, n)
A[1:2, 1:2] = penzl(1)
A[3:n, 3:n] .= -1/2

@testset "Heuristic Penzl" begin
    k = 2
    strategy = Penzl(k, 2, 2)
    shifts = init(strategy, (; E, A))

    @test shifts isa Vector{ComplexF64}
    @test k <= length(shifts) <= k + 1
    @test any(≈(-1 + im), shifts)
    @test any(≈(-1 - im), shifts)

    if length(shifts) > k
        # The strategy may only report more shifts than requested
        # if the last one has been complex, i.e., both the complex
        # parameters as well as its conjugate had to be returned.
        # In that case, the real shift had to be the first one.
        @test any(≈(-1/2), shifts)
        @test shifts[1] ≈ -1/2
    end

    # Ensure complex shifts occur in conjugated pairs:
    i = findfirst(!isreal, shifts)
    @test !isnothing(i)
    @test shifts[i] == conj(shifts[i+1])
end

@testset "Cyclic helper" begin
    shifts = init(Cyclic(1:3), nothing)
    @test take!(shifts) == 1
    @test take!(shifts) == 2
    @test take!(shifts) == 3
    @test take!(shifts) == 1

    shifts = init(Cyclic(Penzl(1, 1, 1)), (; E, A))
    p = take!(shifts)
    if isreal(p)
        @test take!(shifts) == p
    else
        @test take!(shifts) == conj(p)
        @test take!(shifts) == p
    end
end

struct Dummy <: Shifts.Strategy values end
struct DummyIterator values end
Shifts.init(d::Dummy, _) = Shifts.BufferedIterator(DummyIterator(d.values))
Shifts.take_many!(d::DummyIterator) = d.values

# Prerequisite:
@test init(Dummy(nothing), nothing) isa Shifts.BufferedIterator

@testset "Wrapped helper" begin
    shifts = init(Wrapped(reverse, Dummy([1,2,3])), nothing)
    @test shifts isa Shifts.BufferedIterator
    @test shifts.generator isa Shifts.WrappedIterator
    @test shifts.generator.generator isa DummyIterator
    @test take!(shifts) === complex(3.0)
    @test take!(shifts) === complex(2.0)
    @test take!(shifts) === complex(1.0)
end

@testset "Adaptive Kuerschner" begin
    strategy = KuerschnerV(1)
    shifts = init(strategy, (; E, A))

    # Ensure that no shifts have been computed so far:
    @test shifts isa Shifts.BufferedIterator
    @test isempty(shifts.buffer)

    # Pass some initial data:
    Shifts.update!(shifts, LDLᵀ(zeros(n, 0), zeros(0, 0)), ones(n))
    @test isempty(shifts.buffer)

    # As the initial residual was rank one,
    # only one shift should have been computed:
    @test Shifts.take!(shifts) ≈ -5/6
    @test isempty(shifts.buffer)
end
