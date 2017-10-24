#=
kp = 5
th_d= 3
th_b= 0.2
th_r = 0.9

=#

type RegionMergingConfig
    NodeType::DataType
    EdgeType::DataType
    kp::Float64
    thd::Float64
    thb::Float64
    thr::Float64
    small_region_threshold::Integer
end
function MoG(img::Array{T,3}) where T <: Number
	(nr,nc,nch) = size(img)
    grad1, grad2 = imgradients(img, KernelFactors.sobel)
	mog = zeros(nr,nc,nch)
    for i=1:nch
        mog[:,:,i] = magnitude(grad1[:,:,1],grad2[:,:,2])
	end
	return mog
end
function MoG(img::Array{T,2}) where T<: Number
    (nr,nc) = size(img)
    grad1, grad2 = imgradients(img, KernelFactors.sobel)
    return magnitude(grad1,grad2)
end
function MoG(RGB::Array{T1,3}, D::Array{T2,2}) where {T1<:Number, T2<:Number}
	mogRGB   = MoG(RGB)
	mogD     = MoG(D)
	mog =  max.(mogRGB[:,:,1],mogRGB[:,:,2], mogRGB[:,:,3], mogD)
    mog = mog / maximum(mog)
    return mog
end
function MoG(I::Array{T1,2}, D::Array{T2,2}) where T1<:Number where T2<:Number
    mogI = MoG(I)
    mogD = MoG(D)
    mog  = max.(mogI, mogD)
    return mog./maximum(mog)
end
struct Edge{EdgeType}
    data::EdgeType
    node
end
struct Node{NodeDataType, EdgeType}
    index::Integer
    data::NodeDataType
    neighbors::Vector{Edge{EdgeType}}
end
mutable struct Graph{NodeDataType, EdgeType}
    nodes::Dict{Integer,Node{NodeDataType, EdgeType}}
    indices_count::Integer
    edge_weight::Function
    merge_nodes_data::Function
    function Graph{NodeDataType, EdgeType}(edge_weight::Function, merge_nodes_data::Function) where {NodeDataType, EdgeType}
        local nodes = Dict{Integer,Node{NodeDataType, EdgeType}}()
        return new(nodes,0, edge_weight, merge_nodes_data)
    end
end

import Base.getindex
import Base.haskey
function getindex(r::Graph, k::Integer)
    return r.nodes[k].data
end
function haskey(r::Graph, k::Integer)
    return haskey(r.nodes,k)
end

function add_vertex!(r::Graph{D,E}, data::D,neighbors::Vector{Edge{E}}=Edge{E}[]) where  D where E
    r.indices_count+=1;
    r.nodes[r.indices_count] = Node{D,E}(r.indices_count, data, neighbors )
    return r.indices_count
end
function number_of_adjacents(r::Graph{D,E}, i::Integer)  where D where E
    return length(r.nodes[i].neighbors)
end
function adjacents(r::Graph{D,E}, i::Integer) where D where E
    return vec([(edge.node.index, edge.data) for edge in r.nodes[i].neighbors])
end
function remove_connection_to_node!(n::Node{D,E}, i::Integer) where D where E
    filter!(n->n.node.index !=i, n.neighbors)
end
function vertices(r::Graph{D,E}) where D where E
    return collect(keys(r.nodes))
end
function remove_vertex!(r::Graph{D,E}, i::Integer) where D where E
    for a in r.nodes[i].neighbors
        remove_connection_to_node!(a.node, i)
    end
    delete!(r.nodes, i)
end
function connect!(r::Graph{D,E}, i::Integer, j::Integer) where D where E
    local edge = r.edge_weight(r[i],r[j])
    local pos = findfirst(edge->edge.node.index==j,r.nodes[i].neighbors)
    if pos == 0
        push!(r.nodes[i].neighbors, Edge{E}(edge, r.nodes[j]))
    end
    pos = findfirst(edge->edge.node.index==i,r.nodes[j].neighbors)
    if pos==0
        push!(r.nodes[j].neighbors, Edge{E}(edge, r.nodes[i]))
    end
end
function merge_vertices!(r::Graph{D,E}, i::Integer, j::Integer) where D where E
    local node_i = r.nodes[i]
    local node_j = r.nodes[j]
    remove_vertex!(r, j)
    remove_vertex!(r, i)
    local new_data        = r.merge_nodes_data(node_i.data, node_j.data)
    local new_vertex_id   = add_vertex!(r, new_data)
    local new_neighbors   = filter(x->(x.node.index!=i) && (x.node.index!=j),vcat(node_i.neighbors, node_j.neighbors))
    for edge in new_neighbors
        connect!(r, new_vertex_id, edge.node.index)
    end
end
abstract type AbstractRAGNode end
mutable struct RAGEdge
    wd::Float64
    wb::Float64
end
mutable struct RAGNode <: AbstractRAGNode
    w::MvWatsonExponential
    π::Float64
    mean_color::Vector
    boundary::Matrix{Bool}
    pixel_indices::Vector{Integer}
end

"""
```
function boundaries(segmented_image::Matrix; w::Integer=1)
```
Returns the boundary pixels of every region from the `segmented_image`
"""
function boundaries(segmented_image::Matrix; w::Integer=1)
    (nr, nc) = size(segmented_image)
    b        = [Matrix{Bool}(0,0) for i=1:maximum(segmented_image)]
    R        = CartesianRange(size(segmented_image))
    I1, Iend = first(R), last(R)
    for j in unique(segmented_image)
        img_object = segmented_image.==j
        b[j] = dilate(img_object) - erode(img_object)
    end
    return b
end
function edge_weight(Ni::RAGNode, Nj::RAGNode,mog)
        local wd = min(bregman_divergence(Ni.w, Nj.w), bregman_divergence(Nj.w, Ni.w))
        local boundary_intersection = find(Ni.boundary .& Nj.boundary)
        local wi = sum(mog[boundary_intersection]) / (length(boundary_intersection))
        return RAGEdge(wd,wi)
end


function create_node(::Type{RAGNode}, i, RGB, segmented_image, data, bounda) 
    
    (nr,nc, nd) = size(RGB)
    local pixel_indices = find(segmented_image.==i)
    local π             = length(pixel_indices)/(nr*nc)
    local wmean         = vec(mean(data.normal[1][:,pixel_indices],2))
    local N             = MvWatsonExponential(wmean)
    (rows,cols)         = ind2sub((size(RGB,1),size(RGB,2)),pixel_indices)
    local mean_color    =  vec(mean(broadcast_getindex(RGB,rows,cols,collect(1:3)'),1))
    
   return RAGNode(N, π, mean_color, bounda[i], pixel_indices)
end
"""
```
function create_rag(m::Model, RGB, XYZ, segmented_image::Matrix{Int64}, data::RGBDData)
```
Create the region adjacency graph
"""

function create_rag(::Type{NodeType}, ::Type{EdgeType}, RGB, XYZ, segmented_image::Matrix{T}, data,mog) where NodeType where EdgeType where T<:Integer
    (nr,nc, nd) = size(XYZ)
    clusters    = unique(segmented_image)
    rag         = Graph{NodeType,EdgeType}((n1,n2)->edge_weight(n1,n2,mog), merge_data)
    seg_image   = SegmentedImage(segmented_image, collect(1:maximum(segmented_image)), Dict{Int,Real}(), Dict{Int,Int}() )
    G, vert_map = region_adjacency_graph(seg_image,(x,y)->1);

    bounda      = boundaries(segmented_image)
    local vertex_map = Dict{Integer,Integer}()
    # Vertex Creation
    for i in clusters
        local index = add_vertex!(rag, create_node(NodeType, i, RGB, segmented_image, data, bounda))
        vertex_map[i] = index
    end
    for edge in edges(G)
        local i  = edge.src
        local j  = edge.dst
        local Ni = rag[vertex_map[i]]
        local Nj = rag[vertex_map[j]]
        connect!(rag, vertex_map[i],vertex_map[j])
    end
    return rag
end

function candidancy(cfg::RegionMergingConfig, v::T) where T<:AbstractRAGNode
    return  v.w.concentration > cfg.kp
end
function elegibility(cfg::RegionMergingConfig, v::T, r::RAGEdge) where T<:AbstractRAGNode
    return ( r.wb < cfg.thb) && (r.wd < cfg.thd)
end
function should_merge(cfg::RegionMergingConfig, XYZ,  n1::T, n2::T, r::RAGEdge) where T<:AbstractRAGNode
    return candidancy(cfg,n2) && elegibility(cfg, n1, r) && consistency(cfg, XYZ, n1, n2)
end
function consistency(cfg::RegionMergingConfig, XYZ, v::T, r::T) where T<:AbstractRAGNode
    (indices_r, indices_c) = ind2sub((size(XYZ,1),size(XYZ,2)),vcat(v.pixel_indices, r.pixel_indices))
    local points_3d      = broadcast_getindex(XYZ,indices_r,indices_c,collect(1:3)')
    if size(points_3d,1) > 3
        points_3d            = points_3d'
        (B, P, inliers)     = ransacfitplane(points_3d,Inf,false)
        plir                = length(inliers) / length(indices_r)
   return plir > cfg.thr
    else
        return false
    end
end
function merge_data(i::RAGNode, j::RAGNode)
    local new_π         = i.π + j.π
    local new_η         = (i.w.n*i.π + j.w.n*j.π)/(new_π)
    local new_indices   = vcat(i.pixel_indices, j.pixel_indices)
    local new_color     = (i.mean_color + j.mean_color)/2
    local new_boundaries = i.boundary .| j.boundary
    local new_node      = RAGNode(MvWatsonExponential(new_η), new_π, new_color, new_boundaries, new_indices)

end
"""
```
function remove_small_regions(rag,RGB; small_area::Integer=15)
```
Remove segments below a minimum size, assigning them to a neighbor segment with the most similar average color.
"""
function remove_small_regions(r::Graph{NodeType, EdgeType}, seg_image, mog, small_area::Integer=1000) where NodeType where EdgeType
    local prev_number_of_segments = Inf
    local number_of_segments = length(r.nodes)
    while number_of_segments < prev_number_of_segments
        prev_number_of_segments = number_of_segments
        local areas = [(key, length(r[key].pixel_indices)) for key in vertices(r)]
        sort!(areas, by=x->x[2])
        for (k, area) in areas
            if haskey(r,k) && (length(r[k].pixel_indices) < small_area)
                local adjacent_colors = zeros(size(r[k].mean_color,1),number_of_adjacents(r,k))
                local j = 1
                for (i,w) in adjacents(r,k)
                    adjacent_colors[:,j] = r[i].mean_color
                    j=j+1
                end
                local dist  = colwise(Euclidean(), r[k].mean_color,adjacent_colors)

                min_dist, min_neigh = findmin(dist)
                merge_vertices!(r, k, adjacents(r,k)[min_neigh][1])

            end
        end
        number_of_segments = length(r.nodes)
    end

end
"""
```
function merge_parts(cfg::RegionMergingConfig, m::Model, RGB, XYZ, segmented_image::Matrix{<:Integer}, data::RGBDData )
```

"""
function merge_parts(cfg::RegionMergingConfig, RGB, XYZ, segmented_image::Matrix{<:Integer}, data )
    (nr,nc)  = size(segmented_image)
    mog      = MoG(RGB,XYZ[:,:,3])
    rag      = create_rag(cfg.NodeType, cfg.EdgeType, RGB, XYZ, segmented_image, data, mog )
    remove_small_regions(rag, segmented_image, mog, cfg.small_region_threshold)

    last_number_of_nodes = 0
    number_of_nodes = length(rag.nodes)
    while number_of_nodes != last_number_of_nodes
        last_number_of_nodes = number_of_nodes
        for (i, v) in rag.nodes
            local merged = false
            if candidancy(cfg,rag[i])
                local regions  = sort(adjacents(rag,i), by=x->x[2].wd)
                for (r,w) in regions
                    if should_merge(cfg,XYZ, rag[i], rag[r], w)
                        merge_vertices!(rag,i, r)
                        merged = true
                        break
                    end
                end
                if merged
                    break
                end

            end
        end
        number_of_nodes = length(rag.nodes)
    end

    return rag
end

function process_labels(cfg::RegionMergingConfig, RGB, D, labels, xss)

#    labels  = round.(Integer,mapwindow(median!, labels, (3,3)))
    labels = Images.label_components(labels)
    local new_model       = Model(maximum(labels))
    local n           = size(D,1)*size(D,2)
    local rag =  merge_parts(cfg, RGB, D, labels, xss)

    cluster                = similar(labels)
    j=1
    for k in  vertices(rag)
        cluster[rag[k].pixel_indices]=j
        j=j+1
    end

    cluster  = round.(Integer,mapwindow(median!, cluster, (3,3)))
    return cluster
end
