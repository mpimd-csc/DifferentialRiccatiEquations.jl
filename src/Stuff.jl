# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

module Stuff

export restrict

restrict(A::AbstractMatrix, Q) = Q' * A * Q

end