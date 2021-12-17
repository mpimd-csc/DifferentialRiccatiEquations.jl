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
