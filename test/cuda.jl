# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

using CUDA
@assert CUDA.functional(true)

using DifferentialRiccatiEquations
const DRE = DifferentialRiccatiEquations
using .DRE.Shifts: Cyclic, Heuristic
using .DRE.Shifts: Wrapped, Projection, heuristic

using Test
using LinearAlgebra, SparseArrays
using CUDA.CUSPARSE
using IterativeSolvers: cg!
using TimerOutputs: @timeit_debug

# Define necessary overwrites:
@timeit_debug "CG" function Base.:(\)(A::AbstractCuSparseMatrix, B::CuVecOrMat)
    A⁻¹B = zero(B)
    cg!(A⁻¹B, -A, -B)
    A⁻¹B
end

function DRE.orthf(L::CuMatrix)
    F = svd(L)
    Q = F.U
    X = Diagonal(F.S) * F.Vt
    Q, X
end

LinearAlgebra.factorize(X::AbstractCuSparseMatrix) = X

# Assemble system matrices:
n = 10
E = I + 0.1 * sprand(n, n, 1/n)
E = E + E'
A = -I + 0.1 * sprand(n, n, 1/n)
A = A + A'
B = rand(n, 2)
C = rand(3, n)

@assert isposdef(-A)
@assert isposdef(E)

# Assemble initial value:
L = E \ Matrix(C')
D = Matrix(1.0I(size(L, 2)))
@assert E'L ≈ C'

# Move system matrices to GPU:
Ed = CuSparseMatrixCSR(E)
Ad = CuSparseMatrixCSR(A)
Bd = CuMatrix(B)
Cd = CuMatrix(C)
Ld = CuMatrix(L)
Dd = CuMatrix(D)

# Assemble DRE:
tspan = (1.0, 0.0)
nsteps = 10
dt = (tspan[2] - tspan[1]) / nsteps
prob_cpu = GDREProblem(E, A, B, C, LDLᵀ(L, D), tspan)
prob_gpu = GDREProblem(Ed, Ad, Bd, Cd, LDLᵀ(Ld, Dd), tspan)
prob_xpu = GDREProblem(Ed, Ad, Bd, Cd, LDLᵀ(Ld, D), tspan)

# Collect configurations:
drop_complex(shifts) = filter(isreal, shifts) # FIXME: complex shifts should work just fine
heuristic_shifts = (;
    adi_initprev = false,
    adi_kwargs = (;
        shifts = Cyclic(Wrapped(drop_complex, Heuristic(4, 4, 4))),
    )
)
projection_shifts = (;
    adi_initprev = false,
    adi_kwargs = (;
        shifts = Wrapped(heuristic ∘ drop_complex, Projection(2)),
    ),
)

@testset "DRE" begin
    @testset "$desc shifts" for (desc, config) in (
        ("Heuristic", heuristic_shifts),
        ("Projection", projection_shifts),
    )
        sol_cpu = solve(prob_cpu, Ros1(); dt, config...)
        sol_gpu = solve(prob_gpu, Ros1(); dt, config...)
        sol_xpu = solve(prob_xpu, Ros1(); dt, config...)
        @testset "i=$i" for (i, K) in enumerate(sol_cpu.K)
            @test Matrix(sol_gpu.K[i]) ≈ K
            @test Matrix(sol_xpu.K[i]) ≈ K
        end
    end
end
