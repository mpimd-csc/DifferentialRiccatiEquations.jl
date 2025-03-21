# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

struct ShermanMorrisonWoodburySolver
    A⁻¹U
    V
    SOLVER
    solver
end

@timeit_debug "Sherman-Morrison-Woodbury" function CommonSolve.init(
    prob::BlockLinearProblem,
    smw::ShermanMorrisonWoodbury;
)
    prob.A isa LowRankUpdate || error("Not implemented")
    A, α, U, V = prob.A
    B = prob.B
    @unpack ALG, alg = smw

    A⁻¹U = @timeit_debug "solve (sparse)" solve(BlockLinearProblem(A, U), ALG)
    S = α*I + V * A⁻¹U
    ELTYPE = Base.promote_eltype(A, B)
    if B isa AbstractVector
        T = similar(B, ELTYPE, size(V, 1))
    else
        T = similar(B, ELTYPE, size(V, 1), size(B, 2))
    end

    SOLVER = init(BlockLinearProblem(A, B), ALG)
    solver = init(BlockLinearProblem(S, T), alg)
    ShermanMorrisonWoodburySolver(A⁻¹U, V, SOLVER, solver)
end

@timeit_debug "Sherman-Morrison-Woodbury" function CommonSolve.solve!(smw::ShermanMorrisonWoodburySolver)
    @unpack A⁻¹U, V, SOLVER, solver = smw

    A⁻¹B = @timeit_debug "solve (sparse)" solve!(SOLVER)

    mul!(rhs(solver), V, A⁻¹B)
    S⁻¹VA⁻¹B = @timeit_debug "solve (dense)" solve!(solver)

    # X = A⁻¹B - A⁻¹U * S⁻¹VA⁻¹B
    T = eltype(A⁻¹B)
    X = mul!(A⁻¹B, A⁻¹U, S⁻¹VA⁻¹B, -one(T), one(T))
    return X
end

rhs(smw::ShermanMorrisonWoodburySolver) = rhs(smw.SOLVER)

function Base.show(io::IO, ::MIME"text/plain", smw::ShermanMorrisonWoodbury)
    print(io, typeof(smw), "(")
    @unpack ALG, alg = smw
    if alg isa Backslash
        ALG isa Backslash || print(io, ALG)
    else
        ALG isa Backslash || print(io, ALG, ", ", alg)
    end
    print(io, ")")
end
