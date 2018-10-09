module RGBDSegmentation
export clusterize
abstract type RGBDSegmentationAlgorithm end
import ImageSegmentation:
    labels_map

#include("Utils.jl")

struct CDNSegmentedImage{T<:AbstractArray}
    image_indexmap::T
    segment_labels::Vector{Int}
    segment_means::Dict{Int,Vector{Float64}}
    segment_pixel_count::Dict{Int,Int}
end
"""
    img_labeled = labels_map(seg)
Return an array containing the label assigned to each pixel.
"""
labels_map(seg::CDNSegmentedImage) = seg.image_indexmap
include("RGBDProcessing.jl")
include("Datasets/RGBDDataset.jl")


function clusterize(cfg::T, img::CDNImage, k::Integer) where T
    throw("Not implemented")
end
include("Algorithms/JCSA/DepthWMM.jl")
include("Algorithms/JCSA/JCSA.jl")
include("Algorithms/CDNGraph/CDNGraph.jl")
include("Algorithms/GCF/GCF.jl")
end
