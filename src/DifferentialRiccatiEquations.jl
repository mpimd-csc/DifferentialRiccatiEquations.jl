# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module DifferentialRiccatiEquations

using CommonSolve: CommonSolve, solve
using Compat: @something

using LinearAlgebra
using MatrixEquations: lyapc, lyapcs!, utqu!
using UnPack: @unpack
using SparseArrays: SparseArrays,
                    SparseMatrixCSC,
                    AbstractSparseMatrixCSC,
                    issparse,
                    spzeros,
                    spdiagm
using TimerOutputs: @timeit_debug

include("Stuff.jl")

include("Shifts.jl")
include("Callbacks.jl")
using .Callbacks

include("LDLt.jl")
include("LowRankUpdate.jl")
include("lyapunov/types.jl")
include("lyapunov/adi.jl")

include("util/_zeros.jl")
include("util/_diagm.jl")
include("util/_dcat.jl")
include("util/_hcat.jl")
include("util/restrict.jl")

abstract type Algorithm end
struct Ros1 <: Algorithm end
struct Ros2 <: Algorithm end
struct Ros3 <: Algorithm end
struct Ros4 <: Algorithm end

include("riccati/types.jl")
include("riccati/residual.jl")

include("riccati/dense_ros1.jl")
include("riccati/dense_ros2.jl")
include("riccati/dense_ros3.jl")
include("riccati/dense_ros4.jl")

include("riccati/lowrank_ros1.jl")
include("riccati/lowrank_ros2.jl")

include("riccati/newton-adi.jl")

function CommonSolve.solve(
    p::GDREProblem,
    a::Algorithm;
    dt::Real,
    save_state::Bool=false,
    observer=nothing,
    kwargs...,
)
    _solve(
        p,
        a;
        dt=dt,
        save_state=save_state,
        observer=observer,
        kwargs...,
    )
end

export solve
export GDREProblem, Ros1, Ros2, Ros3, Ros4
export GAREProblem, NewtonADI
export GALEProblem, ADI
export LDLᵀ, concatenate!, compress!

end
