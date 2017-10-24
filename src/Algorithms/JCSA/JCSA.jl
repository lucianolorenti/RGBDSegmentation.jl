module JCSA
using ..DepthWMM
import Base.getindex
import Base.setindex!
import Base.length
using Iterators
using Distributions
using ImageFiltering
using ImageProjectiveGeometry
using Images
using LightGraphs
using ImageSegmentation
using PyPlot
include("ExponentialDistributions.jl")
include("combined_kmeans.jl")
import RGBDSegmentation: rgb_plane2rgb_world
import RGBDSegmentation:clusterize
struct Config
    kp::Float64
    thd::Float64
    thb::Float64
    thr::Float64
    max_iterations::Integer

end

"""
Construct a JCSA Config

# Arguments
- `kp::Float64`  Threshold  to decide a region as planar
- `thd::Float64` Threshold to decide the distance among two regions
- `thb::Float64` Threshold to decide the existence of boundary among two regions
- `thr::Float64` Threshold to determine the goodness of a plane fitting.
"""
function Config(; kp::Float64=5.0, thd::Float64=2.0, thb::Float64=3.0, thr::Float64=0.9, max_iterations::Integer=500)
    return Config(kp,thd,thb,thr,max_iterations)
end

"""
https://gist.github.com/manojpandey/f5ece715132c572c80421febebaf66ae
"""

RGBDComponentExponential = Tuple{MvGaussianExponential, MvGaussianExponential, MvWatsonExponential}
const  RGBDComponent = Tuple{MvNormal, MvNormal, MvWatson}

const SufficientStatisticsGaussian  = Tuple{Array{<:AbstractFloat,2},
                                        Array{<:AbstractFloat, 3} }
const SufficientStatisticsNormal = Tuple{Matrix{<:AbstractFloat}}
struct RGBDData
    color::SufficientStatisticsGaussian
    depth::SufficientStatisticsGaussian
    normal::SufficientStatisticsNormal
end
function length(d::RGBDData)
    return size(d.color[1],2)
end
function getindex(d::RGBDData,n::Integer)
    if n==1
        return  d.color
    elseif n==2
        return d.depth
    elseif n==3
        return d.normal
    end
end
function to_source_parameters(p::RGBDComponentExponential)::RGBDComponent
    return  (source_parameters(p[1]),
                          source_parameters(p[2]),
                          source_parameters(p[3] ))
end
import Distributions.pdf
function pdf(c::RGBDComponent, data, i::Integer)::Float64
    prod = pdf(c[1], data.color[1][:,i]) *
           pdf(c[2], data.depth[1][:, i]) *
           pdf(c[3], source_vector(MvWatsonExponential, data.normal[1][:,i]))
    return prod
end
struct Model
    weights::Vector{Float64}
    components::Vector{RGBDComponentExponential}
end
"""
```
function Model(k::Integer)

```
Construct a JCSA Model composed by a mixture of `k` components. Each component is a product of two multivariate gaussian and a multivariate watson distribution
"""
function Model(k::Integer)
    local components = Vector{RGBDComponentExponential}(k)
    for i=1:k
        components[i] = (MvGaussianExponential(3),
                         MvGaussianExponential(1),
                         MvWatsonExponential(3))
    end
    return Model(ones(k)*(1/k), components)
end

include("RegionMerging.jl")

"""
```
function negative_log_likelihood(m::Model, data)::Float64
```
"""
function negative_log_likelihood(m::Model, data)::Float64
    local k = length(m.components)
    local distributions = [to_source_parameters(m.components[i])  for i=1:length(m.components)]
    local n = size(data[1],2)
    local pdfs = [zeros(n,k) for i=1:3]
    for i=1:3
        for j=1:k
            Distributions.logpdf!(view(pdfs[i],:,j), distributions[j][i], data[i])
        end
    end
    logPDF = log.(m.weights)'.+ sum(pdfs)
    y = maximum(logPDF,2);
    x = y.-logPDF
    s = y + log.(sum(exp.(logPDF),2));
    i = find(.!isfinite.(y));
    if !isempty(i)
        s[i]= y[i]
    end
    return -sum(s)
end
import Base.+
"""
```
function expectation(pij::Matrix{Float64}, DG::Array{<:AbstractFloat,3},  m::Model, x::RGBDData)

```
"""
function expectation(pij::Matrix{Float64}, DG::Array{<:AbstractFloat,3},  m::Model, x::RGBDData)
    local k         = length(m.weights)
    local npatterns = length(x)
    local div_sum = 0

    Threads.@threads for j=1:k
    	for r=1:3
            DG[j,r,:] = bregman_divergence(m.components[j][r], x[r]...)
        end
    end

    Threads.@threads for j=1:k
        local dg_k = squeeze(sum(view(DG,j,:,:),1),1)
        pij[j,:] = m.weights[j]*exp.(dg_k);
    end
    pij[isinf.(pij)] = 1e7
    broadcast!(/,pij,pij,sum(pij,1))
end
"""
```
function maximization(component::MvGaussianExponential , pij::Matrix{<:AbstractFloat}, data, j::Integer,sumpij::Vector )
```
Compute the maximization step of a multivariate gaussian
"""
function maximization(component::MvGaussianExponential , pij::Matrix{<:AbstractFloat}, data, j::Integer,sumpij::Vector )
    local dim   = size(data[2],1)
    local n     = size(data[2],3)
    local data2 = reshape(data[2],dim*dim, n)
    local n2    = sum(view(pij,j,:).*data2',1)./sumpij[j]
    n1 = vec(sum(view(pij,j,:).*data[1]',1)./sumpij[j])
    n2 = reshape(n2, (dim,dim))
    component.n =(n1,n2)
end
"""
```
function maximization(component::MvWatsonExponential , pij::Matrix{<:AbstractFloat}, data, j::Integer,sumpij::Vector )
```
Perform the maximization step of the directional data
"""
function maximization(component::MvWatsonExponential , pij::Matrix{<:AbstractFloat}, data, j::Integer,sumpij::Vector )
    component.n =  vec(sum(view(pij,j,:).*data[1]',1)./sumpij[j])
end
"""
```
function maximization(m::Model,pij::Matrix{<:AbstractFloat}, x::RGBDData)
```
"""

function maximization(m::Model,pij::Matrix{<:AbstractFloat}, x::RGBDData)
    local k        = length(m.weights)
    local nvectors = length(x)
    local sumpij   = vec(sum(pij, 2))
    Threads.@threads for j=1:k
        maximization(m.components[j][1], pij, x.color, j, sumpij)
        maximization(m.components[j][2], pij, x.depth, j, sumpij)
        maximization(m.components[j][3], pij, x.normal, j, sumpij)
        m.weights[j] = (sumpij[j]) /nvectors
    end
end

function update_parameters!(m::Model, pij)
	local k = length(m.weights)
    local components_to_remove = []
	local invalid_component_lock = Threads.SpinLock()
	Threads.@threads for i=1:k
        try
            for j=1:3
                update_parameters!(m.components[i][j])
            end
       catch e
		   lock(invalid_component_lock)
           push!(components_to_remove, i)
           warn("Component removed")
 		   unlock(invalid_component_lock)
       end
    end
    deleteat!(m.weights, components_to_remove)
    deleteat!(m.components, components_to_remove)
    for j in reverse(components_to_remove)
        pij = pij[1:end .!= j,: ]
    end
end

function get_extended_data(RGB::Array{<:Number,3}, D::Array{T1,3}, N::Array{T2,3}) where T1<:AbstractFloat where T2<:AbstractFloat
    (nr,nc) = size(D)
    color   = sufficient_statistic(MvGaussianExponential, RGB)
    depth   = sufficient_statistic(MvGaussianExponential, D)
    normal  = sufficient_statistic(MvWatsonExponential, N)
    data    = RGBDData(color,depth,(normal, ))
    return data

end

function hard_clustering( pij::Matrix{<:AbstractFloat}, k::Integer, img_size)
    local labels = ind2sub(size(pij),vec(findmax(pij,1)[2]))[1]
    labels = reshape(labels,img_size[1],img_size[2])
    return  round.(Integer,mapwindow(median!,labels, (3,3)))
end
"""
```
function clusterize(cfg::Config,
                    RGB::Array{<:Number,3},
                    D::Array{<:AbstractFloat,3},
                    N::Array{<:AbstractFloat,3},
                    k::Integer)
```
Entry point of the algorithm.

# Parameters
- `cfg::Config`: Config
- `RGB::Array{<:Number,3}`: Color image of the scene.
- `D::Array{<:AbstractFloat,3}`: 3D image of the scene
- `N::Array{<:AbstractFloat,3}`: Normal vectors image
- `k::Integer`: Number of clust

"""
function clusterize(cfg::Config,
                    RGB::Array{<:Number,3},
                    D::Array{<:AbstractFloat,3},
                    N::Array{<:AbstractFloat,3},
                    k::Integer)
   (model, pij, extended_data) = expectation_maximization(cfg, RGB, D, N, k)
   k      = length(model.components)
   labels = hard_clustering(pij, k, size(D))
   return process_labels(cfg, RGB, D, labels, extended_data)
end
function expectation_maximization(cfg::Config,
                    RGB::Array{<:Number,3},
                    D::Array{<:AbstractFloat,3},
                    N::Array{<:AbstractFloat,3},
                    k::Integer)
	(nr,nc,nch) = size(RGB)
    local nk          = k
    k                 = k
    (assignments, p_data) = initial_clusters(RGB,D,N,k)
    local xss         = get_extended_data(RGB,D,N)
    local model       = Model(k)
    local n           = size(D,1)*size(D,2)
    for i=1:k
        cluster = find(assignments.==i)
        model.weights[i] = length(cluster)/length(assignments)
        model.components[i][1].n = (vec(mean(xss.color[1][:,cluster],2)),
                              squeeze(mean(xss.color[2][:,:,cluster],3)  ,3))

        model.components[i][2].n = (vec(mean(xss.depth[1][:,cluster],2)),
                             squeeze(mean(xss.depth[2][:,:,cluster],3)  ,3))

        model.components[i][3].n = vec(mean(xss.normal[1][:,cluster],2))
	end

    local npatterns = length(xss)
    local pij       = zeros(Float64, k,npatterns)
    update_parameters!(model, pij)
    local prev_nll    = Inf
    local current_nll = negative_log_likelihood(model, p_data)
    local iteration   = 1
    local prev_pij    = nothing
    local DG          = zeros(k,3,npatterns)
    while iteration<cfg.max_iterations && current_nll<prev_nll
        expectation(pij, DG, model, xss)
        maximization(model,pij,xss )

		update_parameters!(model, pij)
        prev_nll    = current_nll
        current_nll = negative_log_likelihood(model, p_data)
        println("Iteracion: $iteration | Mejora NLL: $(prev_nll - current_nll)")
        iteration   = iteration + 1

   end
   return (model,pij, xss)
end
end
