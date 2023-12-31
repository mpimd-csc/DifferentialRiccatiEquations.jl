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

"""
Generalized differential Riccati equation

    E'ẊE = C'C + A'XE + E'XA - E'XBB'XE
    X(t0) = X0

having the fields `E`, `A`, `C`, `X0`, and `tspan`=`(t0, tf)`.
"""
struct GDREProblem{XT}
    E
    A
    B
    C
    X0::XT
    tspan

    GDREProblem(E, A, B, C, X0::XT, tspan) where {XT} = new{XT}(E, A, B, C, X0, tspan)
end

struct DRESolution
    X
    K
    t
end

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

include("riccati/dense_ros1.jl")
include("riccati/dense_ros2.jl")
include("riccati/dense_ros3.jl")
include("riccati/dense_ros4.jl")

include("riccati/sparse_ros1.jl")
include("riccati/sparse_ros2.jl")

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
export LDLᵀ, concatenate!, compress!

end
