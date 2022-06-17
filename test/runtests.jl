using DoubleFloats
using Random
using Setfield
using SparseGridQuadrature
using StaticArrays
using Test

################################################################################

const Dims = 1:4
# const Types = [Float32, Float64, Double64, BigFloat]
const Types = [Float64]
const lmax = 10

################################################################################

const quads = Dict()

@testset "Create quadrature rules T=$T D=$D" for T in Types, D in Dims
    quads[(D, T)] = SGQuadrature{D,T}(lmax)
end

@testset "Basic integration T=$T D=$D" for T in Types, D in Dims
    quad = quads[(D, T)]::SGQuadrature{D,T}

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)

    xmin = ntuple(d -> -1, D)
    xmax = ntuple(d -> +1, D)

    atol = 1 / 2^lmax

    @test quadsg(f0, T, quad, xmin, xmax).result ≈ 2^D
    @test quadsg(f1, T, quad, xmin, xmax).result ≈ 2^D
    @test isapprox(quadsg(f2, T, quad, xmin, xmax).result, D * 2^D; atol=atol)
end

Random.seed!(0)
@testset "Integral bounds T=$T D=$D" for T in Types, D in Dims
    quad = quads[(D, T)]::SGQuadrature{D,T}

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)

    xmin = SVector{D}(-1 + T(rand(-5:5)) / 10 for d in 1:D)
    xmax = SVector{D}(+1 + T(rand(-5:5)) / 10 for d in 1:D)

    atol = 1 / 2^lmax

    @test quadsg(f0, T, quad, xmin, xmax).result ≈ prod(xmax - xmin)
    @test isapprox(quadsg(f1, T, quad, xmin, xmax).result, prod(xmax - xmin) * (1 + sum(xmax + xmin) / 2); atol=atol)
    @test isapprox(quadsg(f2, T, quad, xmin, xmax).result, prod(xmax - xmin) * sum(xmin .^ 2 + xmin .* xmax + xmax .^ 2); atol=atol)
end

Random.seed!(0)
@testset "Vector integrands T=$T D=$D" for T in Types, D in Dims
    quad = quads[(D, T)]::SGQuadrature{D,T}

    xmin = ntuple(d -> -1, D)
    xmax = ntuple(d -> +1, D)

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 .+ b1 * x + b2 * x .^ 2
    g(x) = c0 .+ c1 * x + c2 * x .^ 2

    F = quadsg(f, SVector{D,T}, quad, xmin, xmax).result
    G = quadsg(g, SVector{D,T}, quad, xmin, xmax).result
    @test F isa SVector{D,T}
    @test G isa SVector{D,T}

    afg(x) = a * f(x) + g(x)
    aFG = quadsg(afg, SVector{D,T}, quad, xmin, xmax).result
    @test aFG isa SVector{D,T}
    @test aFG ≈ a * F + G
end

Random.seed!(0)
@testset "Linearity T=$T D=$D" for T in Types, D in Dims
    quad = quads[(D, T)]::SGQuadrature{D,T}

    xmin = ntuple(d -> -1, D)
    xmax = ntuple(d -> +1, D)

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 + b1 * sum(x) + b2 * sum(x .^ 2)
    g(x) = c0 + c1 * sum(x) + c2 * sum(x .^ 2)

    F = quadsg(f, T, quad, xmin, xmax).result
    G = quadsg(g, T, quad, xmin, xmax).result

    afg(x) = a * f(x) + g(x)
    @test quadsg(afg, T, quad, xmin, xmax).result ≈ a * F + G

    @test quadsg(f, T, quad, xmax, xmin).result ≈ (-1)^D * F

    d = T(rand(-9:9)) / 10
    xmin1 = xmin
    xmax1 = @set xmax[1] = d
    xmin2 = @set xmin[1] = d
    xmax2 = xmax
    F1 = quadsg(f, T, quad, xmin1, xmax1).result
    F2 = quadsg(f, T, quad, xmin2, xmax2).result

    atol = 1 / 2^lmax
    @test isapprox(F1 + F2, F; atol=atol)
end
