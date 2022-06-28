using DoubleFloats
using Random
using Setfield
using SparseGridQuadrature
using SpecialFunctions
using StaticArrays
using Test

Random.seed!(0)

################################################################################

const Types = [Float32, Float64, Double64, BigFloat]
Dims(T::Type) = T ∈ [Float32, Float64] ? (1:4) : (1:3)
lmax(D::Int) = 10 + D + (D ≥ 4)

################################################################################

const quads = Dict()

@testset "Create quadrature rules T=$T D=$D" for T in Types, D in Dims(T)
    quads[(D, T)] = SGQuadrature{D,T}(lmax(D))
end

@testset "Basic integration T=$T D=$D" for T in Types, D in Dims(T)
    quad = quads[(D, T)]::SGQuadrature{D,T}

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)
    f3(x) = exp(-sum(x .^ 2))

    # julia> [(quadsg(f2, Float64, SGQuadrature{1,Float64}(lmax)).result - 2, 16 / 4^lmax) for lmax in 1:15]
    # julia> [(quadsg(f2, Float64, SGQuadrature{2,Float64}(lmax)).result - 8, 64 / 4^lmax) for lmax in 1:15]
    # julia> [(quadsg(f2, Float64, SGQuadrature{3,Float64}(lmax)).result - 24, 192 / 4^lmax) for lmax in 1:15]
    # julia> [(quadsg(f2, Float64, SGQuadrature{4,Float64}(lmax)).result - 64, 512 / 4^lmax) for lmax in 1:15]

    # [(abs(quadsg(f3, Float64, SGQuadrature{1,Float64}(lmax)).result / res - 1), 1.31 / 4^lmax) for lmax in 1:15]
    # [(abs(quadsg(f3, Float64, SGQuadrature{2,Float64}(lmax)).result / res^2 - 1), 1.28 * (lmax+2) / 4^lmax) for lmax in 1:15]
    # [(abs(quadsg(f3, Float64, SGQuadrature{3,Float64}(lmax)).result / res^3 - 1), 0.672 * (lmax+2)^2 / 4^lmax) for lmax in 1:15]
    # [(abs(quadsg(f3, Float64, SGQuadrature{4,Float64}(lmax)).result / res^4 - 1), 0.248 * (lmax+2)^3 / 4^lmax) for lmax in 1:15]

    @test quadsg(f0, T, quad).result ≈ 2^D
    @test quadsg(f1, T, quad).result ≈ 2^D
    @test isapprox(quadsg(f2, T, quad).result, D * 2^D; rtol=10 * rtol)
    @test isapprox(quadsg(f3, T, quad).result, (sqrt(T(pi)) * erf(T(1)))^D; rtol=rtol)
end

@testset "Integral bounds T=$T D=$D" for T in Types, D in Dims(T)
    quad = quads[(D, T)]::SGQuadrature{D,T}

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)

    xmin = SVector{D}(-1 + T(rand(-5:5)) / 10 for d in 1:D)
    xmax = SVector{D}(+1 + T(rand(-5:5)) / 10 for d in 1:D)
    quad′ = transform_domain_size!(deepcopy(quad), xmin, xmax)

    @test isapprox(quadsg(f0, T, quad′).result, prod(xmax - xmin); rtol=10 * rtol)
    @test isapprox(quadsg(f1, T, quad′).result, prod(xmax - xmin) * (1 + sum(xmax + xmin) / 2); rtol=rtol)
    @test isapprox(quadsg(f2, T, quad′).result, prod(xmax - xmin) * sum(xmin .^ 2 + xmin .* xmax + xmax .^ 2); rtol=10 * rtol)

    @test isapprox(quadsg(f0, T, quad, xmin, xmax).result, prod(xmax - xmin); rtol=10 * rtol)
    @test isapprox(quadsg(f1, T, quad, xmin, xmax).result, prod(xmax - xmin) * (1 + sum(xmax + xmin) / 2); rtol=rtol)
    @test isapprox(quadsg(f2, T, quad, xmin, xmax).result, prod(xmax - xmin) * sum(xmin .^ 2 + xmin .* xmax + xmax .^ 2);
                   rtol=10 * rtol)
end

@testset "Vector integrands T=$T D=$D" for T in Types, D in Dims(T)
    quad = quads[(D, T)]::SGQuadrature{D,T}

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 .+ b1 * x + b2 * x .^ 2
    g(x) = c0 .+ c1 * x + c2 * x .^ 2

    F = quadsg(f, SVector{D,T}, quad).result
    G = quadsg(g, SVector{D,T}, quad).result
    @test F isa SVector{D,T}
    @test G isa SVector{D,T}

    afg(x) = a * f(x) + g(x)
    aFG = quadsg(afg, SVector{D,T}, quad).result
    @test aFG isa SVector{D,T}
    @test isapprox(aFG, a * F + G; rtol=10 * rtol)
end

@testset "Linearity T=$T D=$D" for T in Types, D in Dims(T)
    quad = quads[(D, T)]::SGQuadrature{D,T}

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 + b1 * sum(x) + b2 * sum(x .^ 2)
    g(x) = c0 + c1 * sum(x) + c2 * sum(x .^ 2)

    F = quadsg(f, T, quad).result
    G = quadsg(g, T, quad).result

    afg(x) = a * f(x) + g(x)
    @test quadsg(afg, T, quad).result ≈ a * F + G

    xmin = SVector{D,T}(-1 for d in 1:D)
    xmax = SVector{D,T}(+1 for d in 1:D)
    quad′ = transform_domain_size!(deepcopy(quad), xmax, xmin)
    @test quadsg(f, T, quad′).result ≈ (-1)^D * F

    d = T(rand(-9:9)) / 10
    xmin1 = xmin
    xmax1 = @set xmax[1] = d
    xmin2 = @set xmin[1] = d
    xmax2 = xmax
    quad1 = transform_domain_size!(deepcopy(quad), xmin1, xmax1)
    quad2 = transform_domain_size!(deepcopy(quad), xmin2, xmax2)
    F1 = quadsg(f, T, quad1).result
    F2 = quadsg(f, T, quad2).result

    @test isapprox(F1 + F2, F; rtol=rtol)
end

################################################################################

@testset "Chebyshev-Gauss: Basic integration T=$T D=$D" for T in Types, D in Dims(T)
    quad = quads[(D, T)]::SGQuadrature{D,T}
    cgquad = transform_chebyshev_gauss!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)
    f3(x) = exp(-sum(x .^ 2))

    # @show abs(quadsg(f0, T, cgquad).result / 2^D - 1)
    # @show abs(quadsg(f1, T, cgquad).result / 2^D - 1)
    # @show abs(quadsg(f2, T, cgquad).result / (D * 2^D) - 1)
    # @show abs(quadsg(f3, T, cgquad).result / (sqrt(T(pi)) * erf(T(1)))^D - 1)
    @test isapprox(quadsg(f0, T, cgquad).result, 2^D; rtol=10 * rtol)
    @test isapprox(quadsg(f1, T, cgquad).result, 2^D; rtol=10 * rtol)
    @test isapprox(quadsg(f2, T, cgquad).result, D * 2^D; rtol=10 * rtol)
    @test isapprox(quadsg(f3, T, cgquad).result, (sqrt(T(pi)) * erf(T(1)))^D; rtol=rtol)
end

@testset "Chebyshev-Gauss: Integral bounds T=$T D=$D" for T in Types, D in Dims(T)
    quad = quads[(D, T)]::SGQuadrature{D,T}
    cgquad = transform_chebyshev_gauss!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)

    xmin = SVector{D}(-1 + T(rand(-5:5)) / 10 for d in 1:D)
    xmax = SVector{D}(+1 + T(rand(-5:5)) / 10 for d in 1:D)
    cgquad′ = transform_domain_size!(deepcopy(cgquad), xmin, xmax)

    @test isapprox(quadsg(f0, T, cgquad′).result, prod(xmax - xmin); rtol=10 * rtol)
    @test isapprox(quadsg(f1, T, cgquad′).result, prod(xmax - xmin) * (1 + sum(xmax + xmin) / 2); rtol=10 * rtol)
    @test isapprox(quadsg(f2, T, cgquad′).result, prod(xmax - xmin) * sum(xmin .^ 2 + xmin .* xmax + xmax .^ 2);
                   rtol=10 * rtol)

    @test isapprox(quadsg(f0, T, cgquad, xmin, xmax).result, prod(xmax - xmin); rtol=10 * rtol)
    @test isapprox(quadsg(f1, T, cgquad, xmin, xmax).result, prod(xmax - xmin) * (1 + sum(xmax + xmin) / 2); rtol=10 * rtol)
    @test isapprox(quadsg(f2, T, cgquad, xmin, xmax).result, prod(xmax - xmin) * sum(xmin .^ 2 + xmin .* xmax + xmax .^ 2);
                   rtol=10 * rtol)
end

@testset "Chebyshev-Gauss: Vector integrands T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    cgquad = transform_chebyshev_gauss!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 .+ b1 * x + b2 * x .^ 2
    g(x) = c0 .+ c1 * x + c2 * x .^ 2

    F = quadsg(f, SVector{D,T}, cgquad).result
    G = quadsg(g, SVector{D,T}, cgquad).result
    @test F isa SVector{D,T}
    @test G isa SVector{D,T}

    afg(x) = a * f(x) + g(x)
    aFG = quadsg(afg, SVector{D,T}, cgquad).result
    @test aFG isa SVector{D,T}
    @test aFG ≈ a * F + G
end

@testset "Chebyshev-Gauss: Linearity T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    cgquad = transform_chebyshev_gauss!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 + b1 * sum(x) + b2 * sum(x .^ 2)
    g(x) = c0 + c1 * sum(x) + c2 * sum(x .^ 2)

    F = quadsg(f, T, cgquad).result
    G = quadsg(g, T, cgquad).result

    afg(x) = a * f(x) + g(x)
    @test quadsg(afg, T, cgquad).result ≈ a * F + G

    xmin = SVector{D,T}(-1 for d in 1:D)
    xmax = SVector{D,T}(+1 for d in 1:D)
    cgquad′ = transform_domain_size!(deepcopy(cgquad), xmax, xmin)
    @test quadsg(f, T, cgquad′).result ≈ (-1)^D * F

    d = T(rand(-9:9)) / 10
    xmin1 = xmin
    xmax1 = @set xmax[1] = d
    xmin2 = @set xmin[1] = d
    xmax2 = xmax
    cgquad1 = transform_domain_size!(deepcopy(cgquad), xmin1, xmax1)
    cgquad2 = transform_domain_size!(deepcopy(cgquad), xmin2, xmax2)
    F1 = quadsg(f, T, cgquad1).result
    F2 = quadsg(f, T, cgquad2).result

    @test isapprox(F1 + F2, F; rtol=rtol)
end

@testset "Chebyshev-Gauss: Integrals of singular functions T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    # Use a double Chebyshev-Gauss transform
    cg2quad = transform_chebyshev_gauss!(deepcopy(quad))
    cg2quad = transform_chebyshev_gauss!(cg2quad)

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f(x) = 1 / sqrt(prod(1 .- x .^ 2))
    F = quadsg(f, T, cg2quad).result
    @test isapprox(F, T(π)^D; rtol=10 * rtol)

    # D=1 is not integrable
    if D ≥ 2
        # Use a double Chebyshev-Gauss transform
        cg2quad = deepcopy(quad)
        cg2quad = transform_chebyshev_gauss!(cg2quad)
        cg2quad = transform_chebyshev_gauss_lb!(cg2quad)
        cg2quad = transform_domain_size!(cg2quad, SVector{D}(0 for d in 1:D), SVector{D}(1 for d in 1:D))

        g(x) = 1 / sqrt(sum(x .^ 2))
        G₀ = T[Inf,
               log(17 + 12 * sqrt(T(2))) / 2,
               -T(π) / 4 - 3 * log(T(2)) / 2 + 3 * log(1 + sqrt(T(3))),
               0.967411988854587774082618732438][D]
        G = quadsg(g, T, cg2quad).result
        @test isapprox(G, G₀; rtol=10 * rtol)
    end
end

################################################################################

@testset "tanh-sinh: Basic integration T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    thquad = transform_tanh_sinh!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)
    f3(x) = exp(-sum(x .^ 2))

    # @show abs(quadsg(f0, T, thquad).result / 2^D - 1)
    # @show abs(quadsg(f1, T, thquad).result / 2^D - 1)
    # @show abs(quadsg(f2, T, thquad).result / (D * 2^D) - 1)
    # @show abs(quadsg(f3, T, thquad).result / (sqrt(T(pi)) * erf(T(1)))^D - 1)
    @test isapprox(quadsg(f0, T, thquad).result, 2^D; rtol=rtol)
    @test isapprox(quadsg(f1, T, thquad).result, 2^D; rtol=rtol)
    @test isapprox(quadsg(f2, T, thquad).result, D * 2^D; rtol=10 * rtol)
    if !((T ≡ Double64 && D ≥ 4) || (T ≡ BigFloat && D ≥ 3))
        # For some reason, high precision high dimensions are rather inaccurate
        @test isapprox(quadsg(f3, T, thquad).result, (sqrt(T(pi)) * erf(T(1)))^D; rtol=100 * rtol)
    end
end

@testset "tanh-sinh: Integral bounds T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    thquad = transform_tanh_sinh!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f0(x) = 1
    f1(x) = sum(x) + 1
    f2(x) = 3 * sum(x .^ 2)

    xmin = SVector{D}(-1 + T(rand(-5:5)) / 10 for d in 1:D)
    xmax = SVector{D}(+1 + T(rand(-5:5)) / 10 for d in 1:D)
    thquad′ = transform_domain_size!(deepcopy(thquad), xmin, xmax)

    @test isapprox(quadsg(f0, T, thquad′).result, prod(xmax - xmin); rtol=rtol)
    @test isapprox(quadsg(f1, T, thquad′).result, prod(xmax - xmin) * (1 + sum(xmax + xmin) / 2); rtol=rtol)
    @test isapprox(quadsg(f2, T, thquad′).result, prod(xmax - xmin) * sum(xmin .^ 2 + xmin .* xmax + xmax .^ 2);
                   rtol=10 * rtol)

    @test isapprox(quadsg(f0, T, thquad, xmin, xmax).result, prod(xmax - xmin); rtol=rtol)
    @test isapprox(quadsg(f1, T, thquad, xmin, xmax).result, prod(xmax - xmin) * (1 + sum(xmax + xmin) / 2); rtol=rtol)
    @test isapprox(quadsg(f2, T, thquad, xmin, xmax).result, prod(xmax - xmin) * sum(xmin .^ 2 + xmin .* xmax + xmax .^ 2);
                   rtol=10 * rtol)
end

@testset "tanh-sinh: Vector integrands T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    thquad = transform_tanh_sinh!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 .+ b1 * x + b2 * x .^ 2
    g(x) = c0 .+ c1 * x + c2 * x .^ 2

    F = quadsg(f, SVector{D,T}, thquad).result
    G = quadsg(g, SVector{D,T}, thquad).result
    @test F isa SVector{D,T}
    @test G isa SVector{D,T}

    afg(x) = a * f(x) + g(x)
    aFG = quadsg(afg, SVector{D,T}, thquad).result
    @test aFG isa SVector{D,T}
    @test aFG ≈ a * F + G
end

@testset "tanh-sinh: Linearity T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    thquad = transform_tanh_sinh!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    a = 1 + T(rand(-5:5)) / 10
    b0 = 1 + T(rand(-5:5)) / 10
    b1 = 1 + T(rand(-5:5)) / 10
    b2 = 1 + T(rand(-5:5)) / 10
    c0 = 1 + T(rand(-5:5)) / 10
    c1 = 1 + T(rand(-5:5)) / 10
    c2 = 1 + T(rand(-5:5)) / 10

    f(x) = b0 + b1 * sum(x) + b2 * sum(x .^ 2)
    g(x) = c0 + c1 * sum(x) + c2 * sum(x .^ 2)

    F = quadsg(f, T, thquad).result
    G = quadsg(g, T, thquad).result

    afg(x) = a * f(x) + g(x)
    @test quadsg(afg, T, thquad).result ≈ a * F + G

    xmin = SVector{D,T}(-1 for d in 1:D)
    xmax = SVector{D,T}(+1 for d in 1:D)
    thquad′ = transform_domain_size!(deepcopy(thquad), xmax, xmin)
    @test quadsg(f, T, thquad′).result ≈ (-1)^D * F

    d = T(rand(-9:9)) / 10
    xmin1 = xmin
    xmax1 = @set xmax[1] = d
    xmin2 = @set xmin[1] = d
    xmax2 = xmax
    thquad1 = transform_domain_size!(deepcopy(thquad), xmin1, xmax1)
    thquad2 = transform_domain_size!(deepcopy(thquad), xmin2, xmax2)
    F1 = quadsg(f, T, thquad1).result
    F2 = quadsg(f, T, thquad2).result

    @test isapprox(F1 + F2, F; rtol=rtol)
end

@testset "tanh-sinh: Integrals of singular functions T=$T D=$D" for T in Types, D in Dims(T)
    # These test cases are too slow
    T ≡ Double64 && D == 4 && continue

    quad = quads[(D, T)]::SGQuadrature{D,T}
    thquad = transform_tanh_sinh!(deepcopy(quad))

    n = 2^(lmax(D) - 1) + 1
    rtol = max(log2(n)^(D - 1) / n^2, sqrt(eps(T)))

    f(x) = 1 / sqrt(prod(1 .- x .^ 2))
    F = quadsg(f, T, thquad).result
    @test isapprox(F, T(π)^D; rtol=10 * rtol)

    # D=1 is not integrable.
    # Float32 is too inaccurate.
    if D ≥ 2 && T ≢ Float32
        thquad = deepcopy(quad)
        thquad = transform_tanh_sinh!(thquad)
        thquad = transform_domain_size!(thquad, SVector{D}(0 for d in 1:D), SVector{D}(1 for d in 1:D))

        g(x) = 1 / sqrt(sum(x .^ 2))
        G₀ = T[Inf,
               log(17 + 12 * sqrt(T(2))) / 2,
               -T(π) / 4 - 3 * log(T(2)) / 2 + 3 * log(1 + sqrt(T(3))),
               0.967411988854587774082618732438][D]
        G = quadsg(g, T, thquad).result
        # Yes, BigFloat is here less accurate than the other types.
        rtol′ = (T ≡ BigFloat ? 200 : 25) * rtol
        @test isapprox(G, G₀; rtol=rtol′)
    end
end
