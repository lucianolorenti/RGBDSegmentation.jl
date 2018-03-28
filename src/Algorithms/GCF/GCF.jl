"""
Fusion of Geometry and Color Information for Scene Segmentation
"""
module GCF
export Config


import RGBDSegmentation:
    clusterize, CDNImage,to_array, rgb2lab!, FRCRGBD, evaluate, color_image, z_image
using RGBDSegmentation
using Images
struct Config
    nev::Integer
    r::Integer
    lambdas
    proportion_of_pixels::Float64
end

using Clustering
using SpectralClustering
using Distances

function weight(i::Integer,j::Vector{<:Integer},pixel_i, neighbors_data)
    local dist_p = Distances.colwise(SqEuclidean(),pixel_i[1:3], neighbors_data[1:3,:])    
    local dist_c = Distances.colwise(SqEuclidean(),pixel_i[4:6], neighbors_data[4:6,:])
    local v =  exp.(-dist_p./(2*mean(dist_p))) .* exp.(-dist_c./(2*mean(dist_c)))
    return v
end
function clusterize(cfg::Config, img_a::CDNImage, k::Integer)
    rgb2lab!(img_a)
    local lab_image = color_image(img_a)
    local d_image = z_image(img_a)
    local best_clus = []
    local cluster_resu = []
    local best_perf    = -Inf
    local curr_perf    = -5000
    local lambda       = 0.05
    for lambda=cfg.lambdas
        println(lambda, " ",curr_perf, " ",best_perf)
        local img           = copy(img_a)
        local img_array     = to_array(img)
        local stds          = vec(std(img_array,2:3))
#        img_array[:,:,1:3] *= (3/sum(stds[1:3]))
#        img_array[:,:,4:6] *= lambda*(3/sum(stds[4:6]))
        local n_of_pixels   = size(img,1)*size(img,2)
        local n_of_samples  = round(Int,n_of_pixels * cfg.proportion_of_pixels)
        local knnconfig     = PixelNeighborhood(cfg.r)
        local nystrom       = NystromMethod(
            EvenlySpacedLandmarkSelection(),            
            n_of_samples,       
            weight,
            cfg.nev,
            true    )

        cluster_resu = SpectralClustering.clusterize(nystrom,KMeansClusterizer(7),img)
        
        local labels    = Images.label_components(reshape(assignments(cluster_resu),size(img_a)))
        labels = round.(Integer,mapwindow(median!, labels, (3,3)))
        gc()
        curr_perf = RGBDSegmentation.evaluate(FRCRGBD(), lab_image, d_image, labels[:])
        if (curr_perf > best_perf)
            best_perf = curr_perf
            best_clus = cluster_resu
        end
        gc()
    end
    return best_clus
end
end
