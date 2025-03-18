# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

@timeit_debug "Sherman-Morrison-Woodbury" function CommonSolve.solve(
    prob::BlockLinearProblem,
    smw::ShermanMorrisonWoodbury;
)
    prob.A isa LowRankUpdate || error("Not implemented")
    A, α, U, V = prob.A
    B = prob.B
    @unpack ALG, alg = smw

    BU = [B U]
    cols = B isa AbstractVector ? 1 : (1:size(B, 2))
    A⁻¹_BU = @timeit_debug "solve (sparse)" solve(BlockLinearProblem(A, BU), ALG)
    A⁻¹B = A⁻¹_BU[:, cols]
    A⁻¹U = A⁻¹_BU[:, last(cols)+1:end]

    S = α*I + V * A⁻¹U
    VA⁻¹B = V * A⁻¹B
    S⁻¹VA⁻¹B = @timeit_debug "solve (dense)" solve(BlockLinearProblem(S, VA⁻¹B), alg)

    # X = A⁻¹B - A⁻¹U * S⁻¹VA⁻¹B
    T = eltype(A⁻¹B)
    X = mul!(A⁻¹B, A⁻¹U, S⁻¹VA⁻¹B, -one(T), one(T))
    return X
end

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
