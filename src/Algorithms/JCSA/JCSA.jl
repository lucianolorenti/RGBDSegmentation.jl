module JCSA
using ..DepthWMM
import Base.getindex
import Base.setindex!
import Base.length
using Distributions
using ImageFiltering
using LightGraphs
using ImageSegmentation
include("ExponentialDistributions.jl")
include("combined_kmeans.jl")
import RGBDSegmentation: rgb_plane2rgb_world
import RGBDSegmentation:clusterize
import RGBDSegmentation: clusterize,
    CDNImage,
    to_array,
    CDNN,
    rgb2lab!,
    FRCCDN,
    evaluate,
    color_image,
    z_image,
    colors,
    distances,
    normals

@enum ComponentType Color=1 Depth=2 Normal=3
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

const CDNComponentExponential = Tuple{MvGaussianExponential,
                                      MvGaussianExponential,
                                      MvWatsonExponential}
const  CDNComponent = Tuple{MvNormal,
                            MvNormal,
                            MvWatson}

const SufficientStatisticsGaussian  = Tuple{Array{<:AbstractFloat,2},
                                        Array{<:AbstractFloat, 3} }
const SufficientStatisticsNormal = Tuple{Matrix{<:AbstractFloat}}
struct CDNData
    color::SufficientStatisticsGaussian
    depth::SufficientStatisticsGaussian
    normal::SufficientStatisticsNormal
end
function length(d::CDNData)
    return size(d.color[1],2)
end
function getindex(d::CDNData,n::ComponentType)
    if n==Color
        return  d.color
    elseif n==Depth
        return d.depth
    elseif n==Normal
        return d.normal
    end
end
"""
    
"""
struct Model
    weights::Vector{Float64}
    components::Vector{CDNComponentExponential}
end
"""
```
function Model(k::Integer)

```
Construct a JCSA Model composed by a mixture of `k` components. Each component is a product of two multivariate gaussian and a multivariate watson distribution
"""
function Model(initial_assignments, xss)
    (_, n) = size(xss.color[1])
    weights = []
    components = []
    for  cluster in initial_assignments
        push!(weights, length(cluster)/n)
        push!(components, (
            MvGaussianExponential(
                xss.color[1][:, cluster]),
            MvGaussianExponential(
                xss.depth[1][:,cluster]),
            MvWatsonExponential(
                vec(mean(xss.normal[1][:,cluster],dims=2)))
        ))
    end
    return Model(weights, components)
end

include("RegionMerging.jl")

"""
```
function negative_log_likelihood(m::Model, data)::Float64
```
    """
function negative_log_likelihood(m::Model, data)::Float64
    k = length(m.components)
    n = size(data,2)
    pdfs = zeros(n,k,3)
    indices = [1:3, 4:6, 7:9]
    for j=1:k
        try
            distributions = to_source_parameters(m.components[j])
            for i=1:3
                Distributions.logpdf!(view(pdfs, :, j, i),
                                      distributions[i],
                                      data[indices[i], :])
            end
        catch e
        end
    end
    try
    logPDF = log.(m.weights)'.+ dropdims(sum(pdfs, dims=3), dims=3)
    y = maximum(logPDF, dims=2);
    x = y.-logPDF
    s = y + log.(sum(exp.(logPDF), dims=2));
    i = findall(.!isfinite.(y));
    if !isempty(i)
        s[i]= y[i]
    end
    return -sum(s)
    catch e
        return 500
    end
end
import Base.+
"""
```
function expectation(pij::Matrix{Float64}, DG::Array{<:AbstractFloat,3},  m::Model, x::CDNData)

```
"""
function expectation(pij::Matrix{Float64}, DG::Array{<:AbstractFloat,3},  m::Model, x::CDNData)
    k = length(m.weights)
    npatterns = length(x)
    div_sum = 0
    for j=1:k, r=1:3
        DG[j,r,:] = bregman_divergence(
            m.components[j][r],
            x[ComponentType(r)]...)
    end
    for j=1:k
        dg_k = dropdims(sum(DG[j,:,:], dims=1), dims=1)
        pij[j,:] = m.weights[j]*exp.(dg_k);
    end
    pij[isinf.(pij)] .= 1e7
    broadcast!(/,pij,pij,sum(pij,dims=1))
end
"""
```
function maximization(component::MvGaussianExponential , pij::Matrix{<:AbstractFloat}, data, j::Integer,sumpij::Vector )
```
Compute the maximization step of a multivariate gaussian
"""
function maximization(component::MvGaussianExponential , pij::Matrix{<:AbstractFloat}, data, j::Integer,sumpij::Vector )
    dim = size(data[2],1)
    n = size(data[2],3)
    data2 = reshape(data[2],dim*dim, n)
    n2 = sum(view(pij,j,:).*data2', dims=1)./sumpij[j]
    n1 = vec(sum(view(pij,j,:).*data[1]', dims=1)./sumpij[j])
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
    component.n =  vec(sum(view(pij,j,:).*data[1]',dims=1)./sumpij[j])
end
"""
```
function maximization(m::Model,pij::Matrix{<:AbstractFloat}, x::CDNData)
```
"""

function maximization(m::Model,pij::Matrix{<:AbstractFloat}, x::CDNData)
    k = length(m.weights)
    nvectors = length(x)
    sumpij = vec(sum(pij, dims=2))
    #=Threads.@threads=# for j=1:k
        maximization(m.components[j][1], pij, x.color, j, sumpij)
        maximization(m.components[j][2], pij, x.depth, j, sumpij)
        maximization(m.components[j][3], pij, x.normal, j, sumpij)
        m.weights[j] = (sumpij[j]) /nvectors
    end
end

function update_parameters!(m::Model, pij)
    k = length(m.weights)
    components_to_remove = []
    invalid_component_lock = Threads.SpinLock()
   #= Threads.@threads=# for i=1:k
       try
           for j=1:3
               println(j)
               update_parameters!(m.components[i][j])
           end
       catch e
	   lock(invalid_component_lock)
           push!(components_to_remove, i)
           @warn("Component removed $e")
 	   unlock(invalid_component_lock)
       end
    end
    deleteat!(m.weights, components_to_remove)
    deleteat!(m.components, components_to_remove)
    for j in reverse(components_to_remove)
        pij = pij[1:end .!= j,: ]
    end
end

function get_extended_data(data::Array{T, 2}) where T<:Number
    color = sufficient_statistic(MvGaussianExponential, data[1:3, :])
    depth = sufficient_statistic(MvGaussianExponential, data[4:6, :])
    normal = sufficient_statistic(MvWatsonExponential, data[7:9, :])
    data = CDNData(color,depth,(normal, ))
    return data

end

function hard_clustering( pij::Matrix{<:AbstractFloat}, k::Integer, img_size)
    labels = [d[1] for d in findmax(pij, dims=1)[2]]
    labels = reshape(labels,img_size[1],img_size[2])
    return  round.(Integer,mapwindow(median!,labels, (3,3)))
end
"""
```
function clusterize(cfg::Config, img_a::CDNImage, k::Integer)
```
Entry point of the algorithm.

# Parameters
- `cfg::Config`: Config
- `img_a::CDNImage`: Color-Distance-Normals image of the scene.
- `k::Integer`: Number of clust

"""
function clusterize(cfg::Config, img::CDNImage, k::Integer)
    (model, pij, extended_data) = expectation_maximization(cfg, img, k)
    k = length(model.components)
    labels = hard_clustering(pij, k, (size(img,2), size(img,3)))
    return labels
    rm_config = RegionMergingConfig()

    return process_labels(rm_config,
                          colors(img),
                          distances(img),
                          labels,
                          extended_data)
end
function expectation_maximization(cfg::Config,
                                  img::CDNImage,
                                  k::Integer) where T <:AbstractArray

    (nch,nr,nc) = size(img)
    npatterns = nr*nc
    p_data = reshape(img,
                     (nch, nr*nc))
    nk = k
    assignments = initial_clusters(p_data,k)
    xss = get_extended_data(p_data)
    model = Model([vec(assignments.==i) for i=1:k], xss)
    pij = zeros(Float64, k,npatterns)
    prev_nll = Inf
    current_nll = negative_log_likelihood(model, p_data)
    iteration = 1
    prev_pij = nothing
    DG = zeros(k,3,npatterns)
    current_nll = 0
    #while iteration<cfg.max_iterations && current_nll<prev_nll
    for i=1:50
        expectation(pij, DG, model, xss)
        maximization(model,pij,xss )
	update_parameters!(model, pij)
        prev_nll    = current_nll
        current_nll = negative_log_likelihood(model, p_data)
        println("Iteracion: $iteration | Mejora NLL: $(prev_nll - current_nll)")
        iteration = iteration + 1
    end
   return (model,pij, xss)
end
end
