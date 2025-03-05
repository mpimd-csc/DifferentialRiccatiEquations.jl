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
struct ADI <: LyapunovSolver end
struct BartelsStewart <: LyapunovSolver end
struct Kronecker <: LyapunovSolver end
