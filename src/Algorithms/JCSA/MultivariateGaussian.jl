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
    theta::Tuple{Vector{Float64},Matrix{Float64}} # Expectation Parameter
    DLNF::Float64
end

function MvGaussianExponential(dim::Integer)
    return MvGaussianExponential((zeros(3),zeros(dim,dim)),
                                 (zeros(3),zeros(dim,dim)),
                                 0.0)
end
function MvGaussianExponential(n::Vector, Σ::Matrix)

   local g = MvGaussianExponential((n,(-Σ + u*u')),
                                 (zeros(3),zeros(dim,dim)),
                                 0.0)
   update_parameters!(g)
   return g
end
function sufficient_statistic(t::Type{MvGaussianExponential}, x::T) where T<:Number
    return sufficient_statistic(t,[x])
end
function sufficient_statistic(::Type{MvGaussianExponential}, x::Vector)
    local nx = convert(Vector{Float64}, x)

    return vcat(nx, reshape(-1*nx*nx',(length(x)^2)))
end

function sufficient_statistic(::Type{MvGaussianExponential}, x::Array{<:AbstractFloat, 2})
    local x1  = zeros(1, size(x,1)*size(x,2))
    local x2  = zeros(1,1,size(x,1)*size(x,2))
    local idx = 1
    for i in CartesianRange((size(x,1),size(x,2)))
        local v     =  x[i[1],i[2]]
        x2[1,1,idx] = -v*v
        x1[:, idx]  = v
        idx+=1
    end
    return (x1, x2)
end
function sufficient_statistic(::Type{MvGaussianExponential}, x::Array{<:AbstractFloat, 3})
    local x1  = zeros(size(x,3), size(x,1)*size(x,2))
    local x2  = zeros(size(x,3),size(x,3),size(x,1)*size(x,2))
    local idx = 1
    for i in CartesianRange((size(x,1),size(x,2)))
        local v =  vec(x[i[1],i[2],:])
        x2[:,:,idx] = -1*v*v'
        x1[:, idx]  = v
        idx+=1
    end
    return (x1, x2)
end
function dimension(g::MvGaussianExponential)
    local c = length(g.n)
    c= round(Integer,(-1+sqrt(1+4*c))/2)
    return c
end
function get_parameters(v::Vector{Float64}, d::Integer)
    local n = v[1:d]
    local H = reshape(v[d+1:end],(d,d))
    return (n,H)
end

"""
From expectation parameters to natural parameters
"""
function update_parameters!(g::MvGaussianExponential)
    local d = dimension(g)
    (n,H) = g.n
    local invH = inv(H)
    g.DLNF =  -0.5 * log(1.00001+ n'*invH*n) - 0.5*log(det(-H)+eps()) - (dimension(g)/2)*log(2*pi*e)

        #Statistical exponential families:  A digest with flash cards
        #Frank Nielsen,  and Vincent Garcia
    local tmp = inv(H+ n*n')
    g.theta = (tmp*(-n), -0.5*tmp)

    check_matrix =  -(H + n*n')
    if !((rank(check_matrix)==3) && isposdef(check_matrix))
        println(rank(check_matrix))
        println(isposdef(check_matrix))
        throw("Invalid covariance")
    end
end
function bregman_divergence(g::MvGaussianExponential, x1::Vector)
    #Simplification and hierarchical representations of mixtures    of exponential families.    V. Garcia, F. Nielsen
    n = g.n
    x = get_parameters(x1, dimension(g))
     return  g.DLNF +  dot(x[1]-n[1], g.theta[1]) + trace((x[2]-n[2])*g.theta[2]')
end
function trace_axb(a, b)
          return dot(view(a,1,:),view(b,:,1)) + dot(view(a,2,:),view(b,:,2)) + dot(view(a,3,:),view(b,:,3))
       end
function bregman_divergence(g::MvGaussianExponential, x1::Matrix, x2::Array{<:AbstractFloat,3})
    local n          = g.n
    local D          = size(x1,2)
    local X1         = x1.-n[1]
    local X2         = (x2.-n[2])
	local trace_term = zeros(D)
    for i=1:D
		trace_term[i] = trace_axb(view(X2,:,:,i),g.theta[2])
	end

    res =  g.DLNF + (vec(g.theta[1]'*X1)) + trace_term
    return res
end

function source_parameters(g::MvGaussianExponential)
    local d = dimension(g)
    n = g.n
    return Distributions.MvNormal(n[1], -(n[2] + n[1]*n[1]'))
end
