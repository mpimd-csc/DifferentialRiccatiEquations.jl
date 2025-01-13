# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

function CommonSolve.solve(prob::GALEProblem, ::BartelsStewart) 
    @unpack E, A, C = prob

    E isa Matrix || (E = Matrix(E))
    A isa Matrix || (A = Matrix(A))
    C isa Matrix || (C = Matrix(C))

    X = lyapc(A', E', C)
    return X
end