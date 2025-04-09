using Test
using DifferentialRiccatiEquations
using DifferentialRiccatiEquations: Shifts
using .Shifts: Cyclic, Heuristic, Projection, Wrapped

function test_hash_stability(builder)
    h = hash(builder())
    @test hash(builder()) == h
end

twice(x) = 2x
shift_builders = [
    () -> Cyclic([1.0]),
    () -> Cyclic(Heuristic(1, 2, 3)),
    () -> Projection(2),
    () -> Cyclic(Wrapped(twice, Projection(2))),
    () -> Cyclic(Wrapped(twice, Heuristic(1, 2, 3))),
]

@testset "Shift Strategy $(bob())" for bob in shift_builders
    test_hash_stability(bob)
end

adi_builders = [() -> ADI(; shifts=bob()) for bob in shift_builders]

@testset "ADI $i" for (i, bob) in enumerate(adi_builders)
    test_hash_stability(bob)
end
