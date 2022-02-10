restrict(A::AbstractMatrix, Q) = Q' * A * Q

function restrict(AUV::LowRankUpdate, Q)
    A, α, U, V = AUV
    restrict(A, Q) + inv(α) * ((Q'U) * (V*Q))
end
