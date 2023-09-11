# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

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

"""
Generalized algebraic (continuous time) algebraic Riccati equation

    Q + A'XE + E'XA - E'XGXE = 0
"""
struct GAREProblem{TG,TQ}
    E
    A
    G::TG
    Q::TQ
end

abstract type AlgebraicRiccatiSolver end
struct NewtonADI <: AlgebraicRiccatiSolver end
