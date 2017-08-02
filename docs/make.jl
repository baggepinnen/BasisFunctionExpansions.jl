using Documenter, BasisFunctionExpansions

makedocs(doctest = false) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    repo = "github.com/baggepinnen/BasisFunctionExpansions.jl.git",
    julia  = "0.6",
    osname = "linux"
)
