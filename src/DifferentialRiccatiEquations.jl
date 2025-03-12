# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module DifferentialRiccatiEquations

using CommonSolve: CommonSolve, solve
using Compat: @something

using Adapt: adapt
using LinearAlgebra
using MatrixEquations: lyapc, lyapcs!, utqu!
using UnPack: @unpack, unpack
using SparseArrays: SparseArrays,
                    SparseMatrixCSC,
                    AbstractSparseMatrix,
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

# Linear (block) vector solvers:
include("blocklinear/types.jl")
include("blocklinear/backslash.jl")

# Linear matrix equations:
include("lyapunov/types.jl")
include("lyapunov/adi.jl")
include("lyapunov/bartels-stewart.jl")
include("lyapunov/kronecker.jl")
include("lyapunov/residual.jl")

# Utilities:
include("util/_zeros.jl")
include("util/_diagm.jl")
include("util/_dcat.jl")
include("util/_hcat.jl")
include("util/restrict.jl")

abstract type Algorithm end
@kwdef struct Ros1 <: Algorithm
    inner_alg = nothing
end
@kwdef struct Ros2 <: Algorithm
    inner_alg = nothing
end
struct Ros3 <: Algorithm end
struct Ros4 <: Algorithm end

# Nonlinear matrix equations:
include("riccati/types.jl")
include("riccati/residual.jl")

include("riccati/dense_ros1.jl")
include("riccati/dense_ros2.jl")
include("riccati/dense_ros3.jl")
include("riccati/dense_ros4.jl")

include("riccati/lowrank_ros1.jl")
include("riccati/lowrank_ros2.jl")

include("riccati/newton.jl")

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
export residual
export GDREProblem, Ros1, Ros2, Ros3, Ros4
export GAREProblem, Newton
export GALEProblem, ADI, BartelsStewart, Kronecker
export LDLᵀ, concatenate!, compress!

end
