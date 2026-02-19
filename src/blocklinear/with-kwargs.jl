# This file is a part of DifferentialRiccatiEquations. License is MIT: https://spdx.org/licenses/MIT.html

"""
    WithKwargs(alg; kwargs...)

Bundle algorithm `alg` and keyword arguments `kwargs` as another algorithm.
Its sole purpose is the following:

```julia
init(_, kw::WithKwargs; kwargs...) = init(_, kw.alg; kw.kwargs..., kwargs...)
```

The `WithKwargs` wrapper is useful, for example,
if `alg` permits passing `reltol` only when a problem instance is known but not as a part of `alg` itself.
That is, when `init(_, alg; reltol)` is allowed while `alg = Alg(; reltol)` is not,
you can use `WithKwargs(alg; reltol)` instead.

Example:

```julia
bicgstab = WithKwargs(
    LinearSolve.KrylovJL_BICGSTAB();
    maxiters = 2n,
    reltol = 1e-10,
)
adi = ADI(ColumnWise(bicgstab))
```
"""
struct WithKwargs
    alg
    kwargs
    WithKwargs(alg; kwargs...) = new(alg, kwargs)
end

function CommonSolve.init(
    prob::Union{BlockLinearProblem,LinearProblem},
    kw::WithKwargs;
    kwargs...,
)
    init(prob, kw.alg; kw.kwargs..., kwargs...)
end
