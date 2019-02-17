module RGBDSegmentation
using ColorTypes
using Statistics
using LinearAlgebra
using StaticArrays
using JLD
export clusterize
abstract type RGBDSegmentationAlgorithm end
import ImageSegmentation:
   SegmentedImage


include("Datasets/RGBDDataset.jl")
include("Processing.jl")

import ImageSegmentation.SegmentedImage
struct ColorDistanceNormalElement{T <: AbstractFloat }<: Colorant{T, 9}
    color::SVector{3, T}
    distance::SVector{3,T}
    normal::SVector{3, T}
end
struct ColorDistanceNormalSerializer{T <: AbstractFloat}
    data::Vector{T}
end

import Base.zero
import Base.show
import Base.+
import Base.- 
function zero(c::Type{ColorDistanceNormalElement})
    return ColorDistanceNormalElement(
        zeros(SVector{3}),
        zeros(SVector{3}),
        zeros(SVector{3}))
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
import Images.channelview
import Images.colorview
function channelview(img::Array{ColorDistanceNormalElement{T}, 2}) where T
    return reshape(reinterpret(T,  img), (9, size(img)...))
end
function colorview(::Type{RGB}, img::Array{ColorDistanceNormalElement{T}, 2}) where T
    return colorview(RGB, @view channelview(img)[1:3, :, :])
end
function colors(img::Array{ColorDistanceNormalElement, 2})
    colors = zeros(3, size(img)...)
    for I in CartesianIndices(size(img))
        colors[:,I] .= img[I].color
    end
    return colors
end


function JLD.readas(serdata::ColorDistanceNormalSerializer)
    return ColorDistanceNormalElement(
        SVector{3}(serdata.data[1:3]),
        SVector{3}(serdata.data[4:6]),
        SVector{3}(serdata.data[7:9]))
end
function JLD.writeas(data::ColorDistanceNormalElement)
    return ColorDistanceNormalSerializer(
        Vector(vcat(data.color, data.distance, data.normal))
    )
end


function SegmentedImage(img::CDNImage, labels::Matrix)
    label_list = convert.(Integer, sort(unique(labels)))
    region_means = Dict{Int, ColorDistanceNormalElement}()
    region_pix_count = Dict{Int, Int}() 
    for label in label_list
        segment = findall(labels .== label)
        region_pix_count[label] = length(segment)
        mean_color = vec(mean(img[1:3, segment], dims=2))
        mean_dist = vec(mean(img[4:6, segment], dims=2))
        mean_normal = sum(img[7:9, segment], dims=2)
        mean_normal = vec(mean_normal / norm(mean_normal))
        region_means[label] = ColorDistanceNormalElement(
            SVector{3,Float64}(mean_color...),
            SVector{3,Float64}(mean_dist...),
            SVector{3,Float64}(mean_normal...))
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
