using Documenter, BasisFunctionExpansions

makedocs()

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    repo = "github.com/baggepinnen/BasisFunctionExpansions.jl.git",
    julia  = "0.6",
    osname = "linux"
)
