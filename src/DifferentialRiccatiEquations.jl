module DifferentialRiccatiEquations

using CommonSolve: CommonSolve, solve

using LinearAlgebra: schur
using MatrixEquations: lyapc, lyapcs!, utqu!
using UnPack: @unpack

struct GDREProblem
    E
    A
    B
    C
    X0
    tspan
end

struct DRESolution
    X
    K
    t
end

abstract type Algorithm end
struct Ros1 <: Algorithm end
struct Ros2 <: Algorithm end
struct Ros3 <: Algorithm end
struct Ros4 <: Algorithm end

include("dense_ros1.jl")
include("dense_ros2.jl")
include("dense_ros3.jl")
include("dense_ros4.jl")

function CommonSolve.solve(
    p::GDREProblem,
    a::Algorithm;
    dt::Real,
    save_state::Bool=false,
)
    _solve(
        p,
        a;
        dt=dt,
        save_state=save_state,
    )
end

export solve
export GDREProblem
export Ros1, Ros2, Ros3, Ros4

end
