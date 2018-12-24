#Pedro F Felzenszwalb and Daniel P Huttenlocher. Efficient graph-based image segmentation.International Journal of Computer Vision, 59(2):167–181, 2004
# Adapted to use RGB
module CDNGraph
export config
import RGBDSegmentation:
    CDNImage,
    clusterize,
    CDNSegmentedImage
import ImageSegmentation:
    felzenszwalb,
    ImageEdge,
    meantype,
    SegmentedImage,
    _colon
using LinearAlgebra
using DataStructures

"""
    edge = ImageEdge(index1, index2, weight)
Construct an edge in a Region Adjacency Graph. `index1` and `index2` are the integers corresponding to individual pixels/voxels (in the sense of linear indexing via `sub2ind`), and `weight` is the edge weight (measures the dissimilarity between pixels/voxels).
"""
struct ImageEdgeCDN
    index1::Int
    index2::Int
    weightRGB::Float64
    weightXYZ::Float64
    weightN::Float64
end
function felzenszwalb(edges::Array{ImageEdgeCDN}, num_vertices::Int, k::Real, min_size::Int = 0)

    num_edges = length(edges)
    G = IntDisjointSets(num_vertices)
    set_size = ones(num_vertices)
    thresholdRGB = fill(convert(Float64,k), num_vertices)
    thresholdXYZ = fill(convert(Float64,k), num_vertices)
    thresholdN   = fill(convert(Float64,k), num_vertices)

    sort!(edges, lt = (x,y)->(x.weightRGB<y.weightRGB))

    for edge in edges
        (wRGB, wXYZ, wN) = (edge.weightRGB, edge.weightXYZ, edge.weightN)
        a = find_root(G, edge.index1)
        b = find_root(G, edge.index2)
        if a!=b
            if (wRGB <= min(thresholdRGB[a], thresholdRGB[b])) && (wXYZ <= min(thresholdXYZ[a], thresholdXYZ[b]))  && (wN <= min(thresholdN[a], thresholdN[b]))
                merged_root = union!(G, a, b)
                set_size[merged_root] = set_size[a] + set_size[b]
                thresholdRGB[merged_root] = wRGB + k/set_size[merged_root]
                thresholdXYZ[merged_root] = wXYZ + k/set_size[merged_root]
                thresholdN[merged_root] = wN + k/set_size[merged_root]
            end
        end
    end

    #merge small segments
    for edge in edges
        a = find_root(G, edge.index1)
        b = find_root(G, edge.index2)
        if a!=b && (set_size[a] < min_size || set_size[b] < min_size)
            union!(G, a, b)
        end
    end

    segments = OrderedSet()
    for i in 1:num_vertices
        push!(segments, find_root(G, i))
    end

    num_sets = length(segments)
    segments2index = Dict{Int, Int}()
    for (i, s) in enumerate(segments)
        segments2index[s]=i
    end

    index_map = Array{Integer}(undef, num_vertices)
    for i in 1:num_vertices
        index_map[i] = segments2index[find_root(G, i)]
    end

    return index_map, num_sets
end
function felzenszwalb(img::CDNImage{T}, k::Real, edge_weight::Function, min_size::Int = 0) where T<:Real
    _, rows, cols = size(img)
    num_vertices = rows*cols
    num_edges = 4*rows*cols - 3*rows - 3*cols + 2
    edges = Array{ImageEdgeCDN}(undef, num_edges)

    R = CartesianIndices((size(img,2),size(img,3)))
    I1, Iend = first(R), last(R)
    num = 1
    for I in R
        for J in CartesianIndices(_colon(max(I1, I-I1), min(Iend, I+I1)))
            if I >= J
                continue
            end
            edges[num] = ImageEdgeCDN(
                (I[2]-1)*rows+I[1],
                (J[2]-1)*rows+J[1],
                edge_weight(img[:,I[1],I[2]],img[:,J[1],J[2]])...)
            num += 1
        end
    end

    index_map, num_segments = felzenszwalb(edges, num_vertices, k, min_size)

    labels = zeros(Integer, (size(img,2),size(img,3)))
    for j=1:size(img, 3), i=1:size(img, 2)
       labels[i, j] = index_map[(j-1)*rows+i]
    end
    return SegmentedImage(img, labels)
end
function weight(a::Vector, b::Vector)
    return (norm(a[1:3]-b[1:3]),norm(a[4:6]-b[4:6]),acos(clamp(dot(a[7:9], b[7:9]),0,1)))
end
function felzenszwalb(img::CDNImage, k::Real, min_size::Int=0)
    return felzenszwalb(img,k, weight, min_size)
end
struct Config
   min_size::Integer
end
function Config(;min_size::Integer)
    return Config(min_size)
end
function clusterize(cfg::Config, img::CDNImage, k::Integer) 
    return felzenszwalb(img, k, cfg.min_size)
end
end
