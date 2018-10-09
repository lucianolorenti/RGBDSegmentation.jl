import Base.-
import Base.getindex
import Base.setindex!
import Base.*
import Base.+
import Base./
import Base.copy
import Base.zero
using Distributions
mutable struct MvGaussianExponential
    n::Tuple{Vector{Float64},Matrix{Float64}} # Expectation Parameter
    gradG::Tuple{Vector{Float64},Matrix{Float64}} 
    G::Float64
end
function MvGaussianExponential(data::Matrix)
    Σ = cov(data, dims=2)
    μ = vec(mean(data, dims=2))
    n = (μ, -(Σ + μ*μ'))
    (G, gradG) = compute_G(n[1], n[2])
    return MvGaussianExponential(
        n,
        gradG,
        G)
end
function sufficient_statistic(::Type{MvGaussianExponential}, x::Vector)
    x1 = zeros(1,size(x,1))
    x2 = zeros(1,1,size(x,1))
    for i=1:size(x,2)
        v = x[i]
        x2[1,1,i] = -v*v'
        x1[1,i] = v
    end
    return (x1, x2)
end

function sufficient_statistic(::Type{MvGaussianExponential}, x::AbstractArray{<:Number,2})
    x1 = zeros(size(x,1), size(x,2))
    x2 = zeros(size(x,1), size(x,1), size(x,2))
    for i=1:size(x,2)
        v = vec(x[:,i])
        x2[:,:,i] = -v*v'
        x1[:,i]  = v
    end
    return (x1, x2)
end
function dimension(g::MvGaussianExponential)
    c = length(g.n)
    c= round(Integer,(-1+sqrt(1+4*c))/2)
    return c
end
function get_parameters(v::Vector{Float64}, d::Integer)
     n = v[1:d]
     H = reshape(v[d+1:end],(d,d))
    return (n,H)
end

function compute_G(n::Vector, H::Matrix)
    d = size(n, 1)
    invH = inv(H)
    G =  -0.5 * real(log(Complex(1.0001+ n'*invH*n))) -
        0.5*log(det(-H)+eps()) -
        (d/2)*log(2*pi*2.71)
    #Satistical exponential families:  A digest with flash cards
    #Frank Nielsen,  and Vincent Garcia
    tmp = inv(H+ n*n')
    gradG = (tmp*(-n), -0.5*tmp)
    return (G, gradG)
end
"""
From expectation parameters to natural parameters
"""
function update_parameters!(g::MvGaussianExponential)
    d = dimension(g)
    (G, gradG) = compute_G(g.n[1], g.n[2])
    g.G = G
    g.gradG = gradG
end
function bregman_divergence(g::MvGaussianExponential, x1::Vector)
    #Simplification and hierarchical representations of mixtures    of exponential families.    V. Garcia, F. Nielsen
    n = g.n
    x = get_parameters(x1, dimension(g))
    return  g.G +  dot(x[1]-n[1], g.gradG[1]) + trace((x[2]-n[2])*g.gradG[2]')
end
function trace_axb(a, b)
    return dot(view(a,1,:),view(b,:,1)) + dot(view(a,2,:),view(b,:,2)) + dot(view(a,3,:),view(b,:,3))
end
function bregman_divergence(g::MvGaussianExponential, x1::Matrix, x2::Array{<:AbstractFloat,3})
    n = g.n
    D = size(x1,2)
    X1 = x1.-n[1]
    X2 = (x2.-n[2])
    trace_term = zeros(D)
    for i=1:D
	trace_term[i] = trace_axb(view(X2,:,:,i),g.gradG[2])
    end
    res =  g.G .+ (vec(g.gradG[1]'*X1)) + trace_term
    return res
end

function expectation_parameters(g::Type{MvGaussianExponential}, data::Matrix)
    μ = vec(mean(data, dims=2))
    Σ = cov(data, dims=2)
    return (μ, -(Σ + μ*μ'))
end
