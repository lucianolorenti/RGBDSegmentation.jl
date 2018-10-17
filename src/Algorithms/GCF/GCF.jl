"""
Fusion of Geometry and Color Information for Scene Segmentation
"""
module GCF
export Config

using Statistics
using RGBDSegmentation
using Images
using ImageSegmentationEvaluation
using ImageSegmentation
import RGBDSegmentation:
    clusterize,
    CDNImage,
    rgb2lab!,
    colors,
    distances,
    number_of_pixels
struct Config
    nev::Integer
    lambdas
    proportion_of_pixels::Float64
    function Config(; nev::Integer=5,
                    lambdas::Array=[0.3,0.4],
                    proportion_of_pixels::Float64=0.001)
        if proportion_of_pixels <0 || proportion_of_pixels >1
            throw("Invalid Proportion")
        end
        return new(nev, lambdas, proportion_of_pixels)
    end
end
using Clustering
using SpectralClustering
using Distances

function weight(i::Integer,
                j::Vector{<:Integer},
                pixel_i,
                neighbors_data)
    dist = Distances.colwise(SqEuclidean(),
                             pixel_i[3:8],
                             neighbors_data[3:8, :])
 
    return exp.(-(dist))
end
function process_labels(labels::SegmentedImage, min_size=350)
    deletion_rule = i -> segment_pixel_count(labels, i) < min_size
    replacement_rule = (i, j) -> -segment_pixel_count(labels, j)
    return prune_segments(labels, deletion_rule, replacement_rule)
end
function clusterize(cfg::Config, img::CDNImage, k::Integer)
    n_of_pixels = number_of_pixels(img)
    best_clus = []
    cluster_resu = []
    best_perf = -Inf
    curr_perf = -5000
    for lambda in cfg.lambdas
        @info "$lambda $curr_perf $best_perf"
        n_of_samples = round(Int,
                             n_of_pixels * cfg.proportion_of_pixels)
        img_a = copy(img)
        rgb2lab!(img_a)
        std_per_dimension = dropdims(std(img_a, dims=[2,3]), dims=(2,3))
        img_a[1:3, :, :] .*= 3/(sum(std_per_dimension[1:3]))
        img_a[4:6, :, :] .*= lambda*(3/(sum(std_per_dimension[4:6])))

        nystrom = NystromMethod(
            EvenlySpacedLandmarkSelection(),            
            n_of_samples,       
            weight,
            cfg.nev,
            true)
        cluster_resu = SpectralClustering.clusterize(
            nystrom,
            KMeansClusterizer(k),
            img_a)

        labels = Images.label_components(
            reshape(
                assignments(cluster_resu),
                (size(img_a, 2), size(img_a, 3))))
        
        segmented_image = SegmentedImage(img, labels)
        labels = process_labels(segmented_image)
        curr_perf = ImageSegmentationEvaluation.evaluate(
            FRCRGBD(),
            colors(img_a),
            z(img_a),
            labels)
        if (curr_perf > best_perf)
            best_perf = curr_perf
            best_clus = segmented_image
        end
    end
    return best_clus
end
end
