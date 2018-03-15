"""
Fusion of Geometry and Color Information for Scene Segmentation
"""
module GCF
export Config


import RGBDSegmentation: clusterize, RGBDNImage,to_array, RGBDN, rgb2lab!


struct Config

end

using Clustering
using SpectralClustering
using Distances
import SpectralClustering: assign!
function assign!(v::T, val::RGBDN) where T<:AbstractArray
    v[:] = val[:]
end

function weight(i::Integer,j::Vector{<:Integer},pixel_i, neighbors_data)
    return vec(prod(exp.(-abs.(pixel_i[3:5].-neighbors_data[3:5,:])./(2*0.2^2)),1))
end
function clusterize(cfg::Config, img::RGBDNImage, k::Integer)
    
    rgb2lab!(img)
    local img_array  = to_array(img)    
    local stds       = vec(std(img_array,2:3))
    img_array[:,:,4:6] *= (3/sum(stds[4:6]))
    img_array[:,:,1:3] *= (3/sum(stds[1:3]))
    
    local number_of_pixels         = size(img,1)*size(img,2)
    local number_of_sampled_points = round(Int,number_of_pixels * 0.01)
    nvec                     = 4
    knnconfig                = PixelNeighborhood(3)
    nystrom                  = NystromMethod(
        EvenlySpacedLandmarkSelection(),
        number_of_sampled_points,
	weight,
        nvec,
        false)
    k                        = nvec
    clustering_result        = SpectralClustering.clusterize(nystrom,YuEigenvectorRotation(),img)       
end
end
