# Run `julia --project=@. --color=yes make.jl` in this directory to
# build the documentation. The `Documenter` must be available (i.e.
# installed globally) for this to work.

using Documenter

using SparseGridQuadrature

makedocs(; sitename="SparseGridQuadrature", format=Documenter.HTML(; prettyurls=false))
deploydocs(; repo="github.com/eschnett/SparseGridQuadrature.jl")
