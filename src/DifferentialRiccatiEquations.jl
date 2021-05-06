module DifferentialRiccatiEquations

import CommonSolve: solve

using MatrixEquations: lyapc
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
    t
end

struct Ros1 end
struct Ros2 end
struct Ros3 end
struct Ros4 end

include("dense_ros1.jl")
include("dense_ros2.jl")

export solve
export GDREProblem
export Ros1, Ros2, Ros3, Ros4

end
