# PkgBenchmark Setup

Ensure the following packages are available in your default load path:

- BenchmarkTools
- MORWiki
- PkgBenchmark

For example, you can add them to your global environment by pasting the following into the REPL:

```
pkg> activate

pkg> add BenchmarkTools MORWiki PkgBenchmark

pkg> activate .

```

To run the whole test suite, start Julia in this project's environment, and execute the following:

```julia
import DifferentialRiccatiEquations
using PkgBenchmark: benchmarkpkg, export_markdown

data = benchmarkpkg(DifferentialRiccatiEquations)

export_markdown(stdout, data)
```

To compare the current state of this project against a baseline, say `HEAD^`, execute the following:

```julia
import DifferentialRiccatiEquations
using PkgBenchmark: judge, export_markdown

data = judge(DifferentialRiccatiEquations, "HEAD^")

export_markdown(stdout, data)
```
