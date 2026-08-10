# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(prob::GALEProblem, ::BartelsStewart)
    (; E, A, C) = prob

    A isa Matrix || (A = Matrix(A))
    C isa Matrix || (C = Matrix(C))

    # Use most optimized method if E matches identity *exactly*.
    E == I && return lyapc(A', C)

    # If E is a multiple of I, move that scaling to the A factor.
    # Mind the adjoint of the scalar.
    if E isa UniformScaling
        # TODO: Scale A in-place if its storage was allocated here.
        A *= E.λ'
        return lyapc(A', C)
    end

    # Fallback: move data to dense CPU storage.
    E isa Matrix || (E = Matrix(E))

    return lyapc(E', A', C)
end
