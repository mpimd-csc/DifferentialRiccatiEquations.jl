module DifferentialRiccatiEquations

import CommonSolve: solve

struct GDREProblem
end

struct DRESolution
end

struct Ros1 end
struct Ros2 end
struct Ros3 end
struct Ros4 end

include("dense_ros1.jl")

export solve
export GDREProblem
export Ros1, Ros2, Ros3, Ros4

end
