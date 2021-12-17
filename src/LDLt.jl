#TODO: add parameter for compression
struct LDLᵀ{TL,TD}
    L::TL
    D::TD

    LDLᵀ(L::TL, D::TD) where {TL, TD} = new{TL,TD}(L, D)
    LDLᵀ{TL,TD}(L::TL, D::TD) where {TL, TD} = new{TL,TD}(L, D)
end

# Destructuring via iteration
Base.iterate(LD::LDLᵀ) = LD.L, :L
Base.iterate(LD::LDLᵀ, _) = LD.D, nothing
