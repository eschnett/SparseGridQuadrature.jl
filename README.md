# SparseGridQuadrature.jl

Multi-dimensional [sparse grid
quadrature](https://en.wikipedia.org/wiki/Sparse_grid) in Julia

* [Documentation](https://eschnett.github.io/SparseGridQuadrature.jl/dev/):
  Future documentation
* [GitHub](https://github.com/eschnett/SparseGridQuadrature.jl): Source
  code repository
* [![GitHub
  CI](https://github.com/eschnett/SparseGridQuadrature.jl/workflows/CI/badge.svg)](https://github.com/eschnett/SparseGridQuadrature.jl/actions)
* [![codecov](https://codecov.io/gh/eschnett/SparseGridQuadrature.jl/branch/main/graph/badge.svg?token=vHtLZhZpKG)](https://codecov.io/gh/eschnett/SparseGridQuadrature.jl)

## Usage

Let us integrate the function `f(x) = 3 x⋅x` over the four-dimensional
cube `[-1; +1]^4`:

```Julia
julia> using SparseGridQuadrature

julia> lmax = 10
10

julia> quad = SGQuadrature{4,Float64}(lmax);

julia> f(x) = 3 * sum(x.^2)
f (generic function with 1 method)

julia> quadsg(f, Float64, quad, (-1,-1,-1,-1), (+1,+1,+1,+1))
(result = 64.00048828125, nevals = 84481)
```

This uses a sparse sampling grid with `2^9` points along each
direction. The Sparse Grid approximation reduces the total number of
grid points from `(2^9)^4 = 2^36 ≈ 7*10^10` points to only `84481`
points, while retaining a comparable level of accuracy.

`SGQuadrature{D,S}(lmax)` creates a sparse grid quadrature method for
`D` dimensions, where coordinates have type `S`. `lmax` is the number
of refinement levels. `L` levels correspond to `2^L+1` points along
each direction. The total number of quadrature points is approximately
`2^D N log(N)^(D-1)`, where `N = 2^(L-1)+1`, and `log` is the binary
(base 2) logarithm.

The second argument to `quadsg`, here `Float64`, describes the return
type of the quadrature.


## References

Thomas Gerstner, Michael Griebel, "Numerical integration using sparse
grids", Numerical Algorithms **18**, 209 (1998), DOI
[10.1023/A:1019129717644](https://doi.org/10.1023/A:1019129717644).

## Related work

Sparse grid quadrature for "Gaussian" integrals:
[SparseGrids.jl](https://github.com/robertdj/SparseGrids.jl)
