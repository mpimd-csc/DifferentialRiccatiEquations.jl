# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using LinearAlgebra
using SparseArrays
using Test

using DifferentialRiccatiEquations
using DifferentialRiccatiEquations: Shifts
using .Shifts
using .Shifts: take!, init

penzl(p) = [-1 p; -p -1]
modified_penzl(v) = abs(real(v)) * penzl(imag(v) / real(v))

n = 3
E = sparse(1.0I(n))
A = spzeros(n, n)
A[1:2, 1:2] = penzl(1)
A[3:n, 3:n] .= -1/2

# Internally, the Ritz values are computed with a naive Arnoldi implementation,
# which is not very accurate. Therefore, the following testset is mostly broken.
# However, in practise, the shifts it produces work much better than accurate Ritz values.
@testset "Heuristic Penzl Shifts" begin
    k = 2
    strategy = Heuristic(k, 2, 2)
    shifts = init(strategy, (; E, A))

    @test shifts isa Vector{ComplexF64}
    @test k <= length(shifts) <= k + 1
    @test_broken any(≈(-1 + im), shifts)
    @test_broken any(≈(-1 - im), shifts)

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
    @test_broken !isnothing(i)
    @test_broken shifts[i] == conj(shifts[i+1])
end

@testset "Cyclic helper" begin
    shifts = init(Cyclic(1:3), nothing)
    @test take!(shifts) == 1
    @test take!(shifts) == 2
    @test take!(shifts) == 3
    @test take!(shifts) == 1

    @testset "Type Stability ($(eltype(values)))" for values in (
        1:2, # iterable
        (1.0, 2.0), # Tuple
        ComplexF64[1, 2], # Vector
    )
        shifts = init(Cyclic(values), nothing)
        a, b = values
        # Check value and type:
        @test take!(shifts) === a
        @test take!(shifts) === b
    end

    shifts = init(Cyclic(Heuristic(1, 1, 1)), (; E, A))
    p = take!(shifts)
    if isreal(p)
        @test take!(shifts) == p
    else
        @test take!(shifts) == conj(p)
        @test take!(shifts) == p
    end

    @testset "Wrapped" begin
        strategy = Cyclic(Wrapped(Base.splat(Returns(42)), Heuristic(1, 1, 1)))
        shifts = init(strategy, (; E, A))
        @test take!(shifts) === 42
        @test take!(shifts) === 42
    end
end

struct Dummy <: Shifts.Strategy values end
struct DummyIterator values end
Shifts.init(d::Dummy, _) = Shifts.BufferedIterator(DummyIterator(d.values))
Shifts.take_many!(d::DummyIterator) = d.values

# Prerequisite:
@test init(Dummy(nothing), nothing) isa Shifts.BufferedIterator

@testset "BufferedIterator helper" begin
    @testset "Type Stability $(eltype(values))" for values in (
        [1, 2, 3],
        ComplexF64[1, 2, 3],
    )
        shifts = init(Dummy(copy(values)), nothing)
        # Check value and type stability:
        for v in values
            @test take!(shifts) === v
        end
    end
end

@testset "Wrapped helper" begin
    shifts = init(Wrapped(reverse, Dummy([1,2,3])), nothing)
    @test shifts isa Shifts.BufferedIterator
    @test shifts.generator isa Shifts.WrappedIterator
    @test shifts.generator.generator isa DummyIterator
    @test take!(shifts) == 3
    @test take!(shifts) == 2
    @test take!(shifts) == 1
end

@testset "Adaptive Projection Shifts" begin
    @test_throws ArgumentError Projection(1)

    strategy = Projection(2)
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

function preserves_conj_pairs(shifts, n=length(shifts); verbose=true)
    i = 0
    while i < n
        i += 1
        v = take!(shifts)
        if !isreal(v)
            i += 1
            w = take!(shifts)
            w ≈ conj(v) && continue
            verbose && @error "Error at shift $i: expected conj($v), got $w"
            return false
        end
    end
    return true
end

# Ensure that complex shifts occur in conjugated pairs.
@testset "Conjugated Pairs" begin
    @testset "Same $desc" for (desc, f) in [
        ("magnitude", a -> -exp(a*im)),
        ("real part", a -> -1 - a*im),
    ]
        @testset "Helper $(length(vals))" for vals in (-1:1, -3:2:3)
            vals = [f(v) for v in -n:2:n]
            @test !preserves_conj_pairs(copy(vals); verbose=false)
            Shifts.safe_sort!(vals)
            @test preserves_conj_pairs(copy(vals))
        end

        @testset "Hacky Projection shifts" begin
            I4 = 1.0 * I(4)
            A = zeros(4, 4)
            A[1:2, 1:2] .= modified_penzl(f(1))
            A[3:4, 3:4] .= modified_penzl(f(2))
            shifts = init(Shifts.Projection(2), (; E=I4, A))

            # Hack input such that full spectrum of A is returned.
            Shifts.update!(shifts, nothing, nothing, I4)
            @test preserves_conj_pairs(shifts, 4)
        end
    end
end
