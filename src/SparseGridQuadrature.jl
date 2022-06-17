module SparseGridQuadrature

using Setfield
using StaticArrays

export SGQuadrature, quadsg

cart(I) = CartesianIndex(Tuple(I))
vect(I) = SVector(Tuple(I))

################################################################################

struct SparseGrid{D,S}
    elts::Array{Array{S,D},D}
    function SparseGrid{D,S}(lmax::Int) where {D,S}
        D::Int
        @assert D ≥ 1
        @assert lmax ≥ 1
        return new{D,S}(Array{Array{S,D}}(undef, ntuple(d -> lmax, D)...))
    end
end

################################################################################

function npoints1d(level::Int)
    @assert level ≥ 1
    return 1 << (level % Unsigned - 1) + 1
end
function weight1d(::Type{S}, level::Int, i::Int) where {S<:Real}
    @assert level ≥ 1
    n = npoints1d(level)
    @assert 1 ≤ i ≤ n
    return S(i == 1 || i == n ? 1 : 2) / (n - 1)
end
function node1d(::Type{S}, level::Int, i::Int) where {S<:Real}
    @assert level ≥ 1
    n = npoints1d(level)
    @assert 1 ≤ i ≤ n
    return S(2i - 1 - n) / (n - 1)
end
"""
    index1d(level::Int, level′::Int, i′::Int)::Int

Find index `i` such that `node1d(T, level, i)` = `node1d(T, level′, i′)`.
"""
function index1d(level::Int, level′::Int, i′::Int)
    @assert level ≥ 1
    n = npoints1d(level)
    @assert level′ ≥ 1
    n′ = npoints1d(level′)
    @assert 1 ≤ i′ ≤ n′
    @assert level ≥ level′      # we require this
    i = ((2i′ - 1 - n′) * ((n - 1) ÷ (n′ - 1)) + n + 1) ÷ 2
    @assert node1d(Float64, level, i) == node1d(Float64, level′, i′)
    return i
end

function sparse_npoints1d(level::Int)
    n = npoints1d(level)
    if level == 1
        n′ = n
    else
        n′ = (n - 1) ÷ 2
    end
    return n′
end
function sparse_to_full1d(level::Int, i::Int)
    n = sparse_npoints1d(level)
    @assert 1 ≤ i ≤ n
    if level == 1
        i′ = i
    else
        i′ = 2i
    end
    n′ = npoints1d(level)
    @assert 1 ≤ i′ ≤ n′
    return i′
end
sparse_node1d(::Type{S}, level::Int, i::Int) where {S<:Real} = node1d(S, level, sparse_to_full1d(level, i))

################################################################################

struct SGQuadrature{D,S}
    lmax::Int
    nodes::SparseGrid{D,SVector{D,S}}
    weights::SparseGrid{D,S}
end

function SGQuadrature{D,S}(lmax::Int) where {D,S<:Real}
    D::Int
    @assert D ≥ 1
    @assert lmax ≥ 1

    nodes = SparseGrid{D,SVector{D,S}}(lmax)
    weights = SparseGrid{D,S}(lmax)

    for grid in cart(ntuple(d -> 1, D)):cart(ntuple(d -> lmax + D - 1, D))
        level = sum(vect(grid))
        if level ≤ lmax + D - 1
            npoints = sparse_npoints1d.(vect(grid))
            # println("grid $(vect(grid))    npoints $(vect(npoints))")

            grid_nodes = Array{SVector{D,S}}(undef, npoints...)
            grid_weights = Array{S}(undef, npoints...)
            for i in cart(ntuple(d -> 1, D)):cart(npoints)
                # println("    i $(vect(i))")

                node = sparse_node1d.(S, vect(grid), vect(i))
                weight = zero(S)
                grid′min = cart(vect(grid) .+ 1)
                grid′max = cart(ntuple(d -> lmax + 2D - 1, D))
                for grid′ in grid′min:grid′max
                    level′ = sum(vect(grid′))
                    if level′ ≤ lmax + 2D - 1
                        q = vect(grid′) - vect(grid)
                        weights1d = SVector{D,S}(begin
                                                     m = vect(grid)[d]
                                                     j = sparse_to_full1d(m, vect(i)[d])
                                                     if q[d] == 1
                                                         weight1d(S, m, j)
                                                     else
                                                         m′ = vect(grid′)[d]
                                                         m1 = m′ - 1
                                                         m2 = m′ - 2
                                                         j1 = index1d(m1, m, j)
                                                         j2 = index1d(m2, m, j)
                                                         weight1d(S, m1, j1) - weight1d(S, m2, j2)
                                                     end
                                                 end
                                                 for d in 1:D)
                        weight += prod(weights1d)
                    end
                end

                grid_nodes[i] = node
                grid_weights[i] = weight
            end
            nodes.elts[grid] = grid_nodes
            weights.elts[grid] = grid_weights
        end
    end

    return SGQuadrature{D,S}(lmax, nodes, weights)
end

################################################################################

function quadsg(f, ::Type{T}, quad::SGQuadrature{D,S}) where {T,D,S<:Real}
    D::Int
    @assert D ≥ 1

    result = one(S) * zero(T)
    nevals = 0

    lmax = quad.lmax
    for grid in cart(ntuple(d -> 1, D)):cart(ntuple(d -> lmax + D - 1, D))
        level = sum(vect(grid))
        if level ≤ lmax + D - 1
            grid_nodes = quad.nodes.elts[grid]
            grid_weights = quad.weights.elts[grid]
            result += sum(grid_weights[i] * T(f(grid_nodes[i])) for i in eachindex(grid_weights))
            nevals += length(grid_weights)
        end
    end

    return (result=result, nevals=nevals)
end

function quadsg(f, ::Type{T}, quad::SGQuadrature{D,S}, xmin::SVector{D,S}, xmax::SVector{D,S}) where {T,D,S<:Real}
    x₀ = (xmin + xmax) / 2
    Δx = (xmax - xmin) / 2
    g(x) = prod(Δx) * f(x₀ + Δx .* x)
    return quadsg(g, T, quad)
end

function quadsg(f, ::Type{T}, quad::SGQuadrature{D,S}, xmin::Union{Tuple,SVector{D}},
                xmax::Union{Tuple,SVector{D}}) where {T,D,S<:Real}
    return quadsg(f, T, quad, SVector{D,S}(xmin), SVector{D,S}(xmax))
end

end
