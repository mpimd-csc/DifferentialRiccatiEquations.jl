# Oberwolfach Rail

using DifferentialRiccatiEquations
using MAT, UnPack, Test

const DREs = DifferentialRiccatiEquations

# Setup
P = matread("Rail371.mat")
@unpack E, A, B, C, X0 = P
Ed = collect(E) # d=dense
tspan = (4500., 0.) # backwards in time
prob = GDREProblem(Ed, A, B, C, X0, tspan)

# Solve
alg = @isdefined(alg) ? alg : Ros1()
sol = solve(prob, alg; dt=-1500)

@test sol isa DREs.DRESolution
