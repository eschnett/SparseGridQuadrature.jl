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

julia> quadsg(f, Float64, quad)
(result = 64.00048828125, nevals = 84481)
```

This uses a sparse sampling grid with `2^9` points along each
direction. The Sparse Grid approximation reduces the total number of
grid points from `(2^9)^4 = 2^36 ≈ 7*10^10` points to only `84481`
points, while retaining a comparable level of accuracy.

`SGQuadrature{D,S}(lmax)` creates a sparse grid quadrature method for
`D` dimensions, where coordinates have type `S`. `lmax` is the number
of refinement levels. `lmax` levels correspond to `N = 2^(lmax-1)+1`
points along each direction.

The total number of quadrature points is approximately `N
log2(N)^(D-1)`, and the expected relative error is approximately
`log2(N)^(D-1) / N^2`. That is, up to logarithmic terms, the total
number of points is of the same order as the number of points along
each edge, and convergence is quadratic.

The second argument to `quadsg`, here `Float64`, describes the return
type of the quadrature.

## Specifying the quadrature domain

If the quadrature domain is not the unit cube, then the quadrature
method `quad` should be updated:

```Julia
julia> using SparseGridQuadrature

julia> lmax = 10
10

julia> quad = SGQuadrature{4,Float64}(lmax);

julia> transform_domain_size!(quad, (0,0,0,0), (1,1,1,1));

julia> f(x) = 3 * sum(x.^2)
f (generic function with 1 method)

julia> quadsg(f, Float64, quad)
(result = 4.000007629394531, nevals = 84481)
```

## Chebyshev-Gauss quadrature

The quadrature method can also be updated to use a [Chebyshev-Gauss
quadrature](https://en.wikipedia.org/wiki/Chebyshev–Gauss_quadrature)
instead of the standard trapezoidal rule:

```Julia
julia> using SparseGridQuadrature

julia> lmax = 14
14

julia> quad = SGQuadrature{4,Float64}(lmax);

julia> transform_domain_size!(quad, (0,0,0,0), (1,1,1,1));

julia> transform_chebyshev_gauss!(quad);

julia> f(x) = 3 * sum(x.^2)
f (generic function with 1 method)

julia> quadsg(f, Float64, quad)
(result = 3.9386494466298867, nevals = 178177)
```

Unfortunately this is not usually more accurate than the trapezoidal
rule. I assume the reason is that a sparse grid quadrature already
emphasizes the boundary points, which already captures much of the
advantage of the Chebyshev-Gauss rules which also clusters points near
the boundary. Also, switching to a Chebyshev-Gauss rule then spreads
out the points in the interior requiring more points overall (a larger
`lmax`).

## Tanh-sinh quadrature

The quadrature method can also be updated to use a [tanh-sinh
quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature)
instead of the standard trapezoidal rule. This is useful when the
integrand is singular at the boundary:

```Julia
julia> using SparseGridQuadrature

julia> lmax = 14
14

julia> quad = SGQuadrature{4,Float64}(lmax);

julia> transform_tanh_sinh!(quad);

julia> f(x) = 1 / sqrt(prod(1 .- x .^ 2))
f (generic function with 1 method)

julia> quadsg(f, Float64, quad)
(result = 97.40908811445765, nevals = 178177)
```

The expected result is `π^4 = 97.40909103400243`, the quadrature error
is `3⋅10^-8`.


## References

Thomas Gerstner, Michael Griebel, "Numerical integration using sparse
grids", Numerical Algorithms **18**, 209 (1998), DOI
[10.1023/A:1019129717644](https://doi.org/10.1023/A:1019129717644).

## Related work

Sparse grid quadrature for "Gaussian" integrals:
[SparseGrids.jl](https://github.com/robertdj/SparseGrids.jl)
