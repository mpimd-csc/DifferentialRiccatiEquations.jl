# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
Generalized algebraic Lyapunov equation

    A'XE + E'XA = -C

having the fields `A`, `E` and `C`.
"""
struct GALEProblem{T}
    E
    A
    C::T

    GALEProblem(E, A, C::T) where {T} = new{T}(E, A, C)
end

abstract type LyapunovSolver end

@kwdef struct ADI <: LyapunovSolver
    maxiters::Int = 100
    reltol::Union{Nothing,Real} = nothing
    abstol::Union{Nothing,Real} = nothing
    shifts::Shifts.Strategy = Shifts.Projection(2)
    ignore_initial_guess::Bool = false # use zero if true
    inner_alg::BlockLinearSolver = Backslash() # to solve BlockLinearProblem
end

ADI(inner_alg; kwargs...) = ADI(; inner_alg, kwargs...)

struct BartelsStewart <: LyapunovSolver end
struct Kronecker <: LyapunovSolver end
