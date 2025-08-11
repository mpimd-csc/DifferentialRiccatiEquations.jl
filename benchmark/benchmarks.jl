using BenchmarkTools
using LinearAlgebra

using MORWiki: assemble, SteelProfile

using DifferentialRiccatiEquations
using DifferentialRiccatiEquations.Shifts

SUITE = BenchmarkGroup()

# Ensure headless downloads succeed:
ENV["DATADEPS_ALWAYS_ACCEPT"] = 1

# Newton method for GARE:
adi = ADI(;
    maxiters=200,
    ignore_initial_guess=true,
    shifts=Cyclic(Heuristic(20, 30, 30)),
)
t = 15
gmres = GMRES(;
    maxiters=5,
    maxrestarts=0,
    ignore_initial_guess=true,
    preconditioner=ADI(;
        maxiters=t,
        shifts=Cyclic(Heuristic(t, t, t)),
        compression_interval=2t,
        warn_convergence=false,
    ),
)
for (desc, inner_solver) in (
    ("adi", adi),
    ("gmres", gmres),
)
    suite = SUITE["newton"][desc]
    newton = Newton(inner_solver;
        maxiters=20,
    )
    for n in (1357, 5177)
        system = SteelProfile(n)
        (; E, A, B, C) = assemble(system)
        B = Matrix(B)
        C = Matrix(C)
        G = lowrank(1000B, I)
        Q = lowrank(C', I)
        are = GAREProblem(E, A, G, Q)
        suite[system] = @benchmarkable solve($are, $newton)
    end
end
