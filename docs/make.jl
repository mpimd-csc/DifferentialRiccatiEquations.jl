using DifferentialRiccatiEquations
using Documenter

DocMeta.setdocmeta!(DifferentialRiccatiEquations, :DocTestSetup, :(using DifferentialRiccatiEquations); recursive=true)

makedocs(;
    modules=[DifferentialRiccatiEquations],
    authors="Jonas Schulze <jschulze@mpi-magdeburg.mpg.de> and contributors",
    repo="https://gitlab.mpi-magdeburg.mpg.de/jschulze/DifferentialRiccatiEquations.jl/blob/{commit}{path}#{line}",
    sitename="DifferentialRiccatiEquations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jschulze.pages.mpi-magdeburg.mpg.de/DifferentialRiccatiEquations.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
