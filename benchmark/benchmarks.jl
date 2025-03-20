using BenchmarkTools
using LinearAlgebra

using MORWiki: assemble, SteelProfile

using DifferentialRiccatiEquations
using DifferentialRiccatiEquations.Shifts

SUITE = BenchmarkGroup()

# Ensure headless downloads succeed:
ENV["DATADEPS_ALWAYS_ACCEPT"] = 1

# Newton method for GARE:
suite = SUITE["newton"]["adi"]
adi_kwargs = (;
    maxiters=200,
    shifts=Cyclic(Heuristic(20, 30, 30)),
)
newton_kwargs = (;
    maxiters=20,
    adi_initprev=false,
    adi_kwargs,
)
for n in (1357, 5177)
    system = SteelProfile(n)
    (; E, A, B, C) = assemble(system)
    B = Matrix(B)
    C = Matrix(C)
    G = LDLᵀ(1000B, I)
    Q = LDLᵀ(C', I)
    are = GAREProblem(E, A, G, Q)
    suite[system] = @benchmarkable solve($are, NewtonADI(); $newton_kwargs...)
end
