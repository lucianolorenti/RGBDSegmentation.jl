module RGBDSegmentation
using ColorTypes
using Statistics
using LinearAlgebra
export clusterize
abstract type RGBDSegmentationAlgorithm end
import ImageSegmentation:
   SegmentedImage


include("Datasets/RGBDDataset.jl")
include("Processing.jl")

import ImageSegmentation.SegmentedImage
struct ColorDistanceNormalElement{T <: AbstractFloat }<: Colorant{T, 9}
    color:: Vector{T}
    distance:: Vector{T}
    normal:: Vector{T}
end
import Base.zero
import Base.show
import Base.+
import Base.- 
function zero(c::Type{ColorDistanceNormalElement})
    return ColorDistanceNormalElement(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
end
function show(io::IO, elem::ColorDistanceNormalElement)
    show(io, elem.color)
    show(io, elem.distance)
    show(io, elem.normal)
end
function -(v1::ColorDistanceNormalElement, v2::ColorDistanceNormalElement)
    n = v1.normal - v2.normal
    n = n / norm(n)
    return ColorDistanceNormalElement(
        v1.color - v2.color,
        v1.distance - v2.distance,
        n)
end
function +(v1::ColorDistanceNormalElement, v2::ColorDistanceNormalElement)
    n = v1.normal + v2.normal
    n = n / norm(n)
    return ColorDistanceNormalElement(
        v1.color + v2.color,
        v1.distance + v2.distance,
        n)
end
function /(v1::ColorDistanceNormalElement, v::Integer)
    n = v1.normal / v
    n = n / norm(n)
    return ColorDistanceNormalElement(
        v1.color / v,
        v1.distance / v,
        n)
end

function SegmentedImage(img::CDNImage, labels::Matrix)
    label_list = sort(unique(labels))
    region_means        = Dict{Int, ColorDistanceNormalElement}()
    region_pix_count    = Dict{Int, Int}() 

    for label in label_list
        segment = findall(labels .== label)
        region_pix_count[label] = length(segment)
        mean_color = vec(mean(img[1:3, segment], dims=2))
        mean_dist = vec(mean(img[4:6, segment], dims=2))
        mean_normal = sum(img[7:9, segment], dims=2)
        mean_normal = vec(mean_normal / norm(mean_normal))
        region_means[label] = ColorDistanceNormalElement(
            mean_color,
            mean_dist,
            mean_normal)
        
        
    end
    return SegmentedImage(labels,
                          label_list,
                          region_means,
                          region_pix_count)
end

# Datasets
include("Datasets/NYUDepthV2.jl")

function clusterize(cfg::T, img::CDNImage, k::Integer) where T
    throw("Not implemented")
end

# Algorithms 
include("Algorithms/DepthWMM/DepthWMM.jl")
include("Algorithms/JCSA/JCSA.jl")
include("Algorithms/CDNGraph/CDNGraph.jl")
include("Algorithms/GCF/GCF.jl")
end
