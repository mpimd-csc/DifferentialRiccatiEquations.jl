module DifferentialRiccatiEquations

import CommonSolve: solve

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

struct Ros1 end
struct Ros2 end
struct Ros3 end
struct Ros4 end

include("dense_ros1.jl")
include("dense_ros2.jl")
include("dense_ros3.jl")
include("dense_ros4.jl")

export solve
export GDREProblem
export Ros1, Ros2, Ros3, Ros4

end
