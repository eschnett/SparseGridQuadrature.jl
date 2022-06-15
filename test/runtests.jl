using DoubleFloats
using Random
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

# Random.seed!(0)
# @testset "Linearity T=$T" for T in Types
#     quad = quads[T]::SGQuadrature{T}
# 
#     a = 1 + T(rand(-5:5)) / 10
#     b0 = 1 + T(rand(-5:5)) / 10
#     b1 = 1 + T(rand(-5:5)) / 10
#     b2 = 1 + T(rand(-5:5)) / 10
#     c0 = 1 + T(rand(-5:5)) / 10
#     c1 = 1 + T(rand(-5:5)) / 10
#     c2 = 1 + T(rand(-5:5)) / 10
# 
#     f(x) = b0 + b1 * x + b2 * x^2
#     g(x) = c0 + c1 * x + c2 * x^2
# 
#     F = quadts(f, quad, -1, +1).result
#     G = quadts(g, quad, -1, +1).result
# 
#     afg(x) = a * f(x) + g(x)
#     @test quadts(afg, quad, -1, +1).result ≈ a * F + G
# 
#     d = T(rand(-9:9)) / 10
# 
#     @test quadts(f, quad, +1, -1).result ≈ -F
# 
#     F0 = quadts(f, quad, -1, a).result
#     F1 = quadts(f, quad, a, +1).result
#     @test F0 + F1 ≈ F
# end
# 
# Random.seed!(0)
# @testset "Integrals of singular functions T=$T" for T in Types
#     quad = quads[T]::SGQuadrature{T}
# 
#     rtol = Dict(Float32 => 10 * sqrt(eps(T)),
#                 Float64 => 10 * sqrt(eps(T)),
#                 Double64 => 10^5 * sqrt(eps(T)),
#                 BigFloat => 1.0e-21)[T]
# 
#     f(x) = 1 / sqrt(1 - x^2)
#     F = quadts(f, quad, -1, +1; rtol=eps(T)^(T(3) / 4)).result
#     @test isapprox(F, T(π); rtol=rtol)
# 
#     rtol = Dict(Float32 => 10 * sqrt(eps(T)),
#                 Float64 => 100 * sqrt(eps(T)),
#                 Double64 => 10^7 * sqrt(eps(T)),
#                 BigFloat => 1.0e-18)[T]
# 
#     a = T(rand(1:9)) / 10
#     g(x) = 1 / x^(1 - a)
#     G = quadts(g, quad, 0, 1; rtol=eps(T)^(T(3) / 4)).result
#     @test isapprox(G, 1 / a; rtol=rtol)
# end
# 
# ################################################################################
# 
# # const Dims = 1:3
# # const nlevels = 8
# # const quadsnd = Dict()
# # 
# # @testset "Create quadrature rules D=$D T=$T" for D in Dims, T in Types
# #     quadsnd[(D, T)] = SGQuadratureND{D,T}(nlevels)
# # end
# # 
# # @testset "Basic N-dimensional integration D=$D T=$T" for D in Dims, T in Types
# #     quad = quadsnd[(D, T)]::SGQuadratureND{D,T}
# # 
# #     f0(x...) = 1
# #     # f1(x...) = sum(x) + 1
# #     # f2(x...) = 3 * sum(x .^ 2)
# # 
# #     @test quadts(f0, quad, ntuple(d -> -1, D), ntuple(d -> +1, D)).result ≈ 2^D
# #     # @test quadts(f1, quad, ntuple(d -> -1, D), ntuple(d -> +1, D)).result ≈ 2^D
# #     # @test quadts(f2, quad, ntuple(d -> -1, D), ntuple(d -> +1, D)).result ≈ 2^D
# # end
# 
# # Random.seed!(0)
# # @testset "Integral bounds T=$T" for T in Types
# #     quad = quads[T]::SGQuadrature{T}
# # 
# #     f0(x) = 1
# #     f1(x) = x + 1
# #     f2(x) = 3 * x^2
# # 
# #     xmin = -1 + T(rand(-5:5)) / 10
# #     xmax = +1 + T(rand(-5:5)) / 10
# # 
# #     @test quadts(f0, quad, xmin, xmax).result ≈ xmax - xmin
# #     @test quadts(f1, quad, xmin, xmax).result ≈ (xmax^2 - xmin^2) / 2 + xmax - xmin
# #     @test quadts(f2, quad, xmin, xmax).result ≈ (xmax^3 - xmin^3)
# # end
# # 
# # Random.seed!(0)
# # @testset "Linearity T=$T" for T in Types
# #     quad = quads[T]::SGQuadrature{T}
# # 
# #     a = 1 + T(rand(-5:5)) / 10
# #     b0 = 1 + T(rand(-5:5)) / 10
# #     b1 = 1 + T(rand(-5:5)) / 10
# #     b2 = 1 + T(rand(-5:5)) / 10
# #     c0 = 1 + T(rand(-5:5)) / 10
# #     c1 = 1 + T(rand(-5:5)) / 10
# #     c2 = 1 + T(rand(-5:5)) / 10
# # 
# #     f(x) = b0 + b1 * x + b2 * x^2
# #     g(x) = c0 + c1 * x + c2 * x^2
# # 
# #     F = quadts(f, quad, -1, +1).result
# #     G = quadts(g, quad, -1, +1).result
# # 
# #     afg(x) = a * f(x) + g(x)
# #     @test quadts(afg, quad, -1, +1).result ≈ a * F + G
# # 
# #     d = T(rand(-9:9)) / 10
# # 
# #     @test quadts(f, quad, +1, -1).result ≈ -F
# # 
# #     F0 = quadts(f, quad, -1, a).result
# #     F1 = quadts(f, quad, a, +1).result
# #     @test F0 + F1 ≈ F
# # end
# # 
# # Random.seed!(0)
# # @testset "Integrals of singular functions T=$T" for T in Types
# #     quad = quads[T]::SGQuadrature{T}
# # 
# #     rtol = Dict(Float32 => 10 * sqrt(eps(T)),
# #                 Float64 => 10 * sqrt(eps(T)),
# #                 Double64 => 10^5 * sqrt(eps(T)),
# #                 BigFloat => 1.0e-21)[T]
# # 
# #     f(x) = 1 / sqrt(1 - x^2)
# #     F = quadts(f, quad, -1, +1; rtol=eps(T)^(T(3) / 4)).result
# #     @test isapprox(F, T(π); rtol=rtol)
# # 
# #     rtol = Dict(Float32 => 10 * sqrt(eps(T)),
# #                 Float64 => 100 * sqrt(eps(T)),
# #                 Double64 => 10^7 * sqrt(eps(T)),
# #                 BigFloat => 1.0e-18)[T]
# # 
# #     a = T(rand(1:9)) / 10
# #     g(x) = 1 / x^(1 - a)
# #     G = quadts(g, quad, 0, 1; rtol=eps(T)^(T(3) / 4)).result
# #     @test isapprox(G, 1 / a; rtol=rtol)
# # end
# 
