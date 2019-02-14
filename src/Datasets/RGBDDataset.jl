export 
    RGBDDataset,
    NYUDataset,
    get_image,
    CDN,
    CDNImage,
    CDNColorant,
    LabelledCDNImage,
    color_image,
    z_image,
    to_array,
    to_rgbdnimage,
    to_image,
    rgb2lab!,
    preprocess,
    colors,
    distances,
    normals,
    z,
    name
import Base.download
using FITSIO
using MAT
using ..RGBDSegmentation
using ColorTypes
using StaticArrays
import Base: show, +,-,*,/,getindex, setindex!,zeros, zero
import ColorTypes: comp1, comp2, comp3
import Images: accum

abstract type RGBDDataset end
import Base.getindex
function getindex(d::RGBDDataset, i::Integer)
    return get_image(d, i)
end

const CDNImage{T} = Array{T, 3} where T<:Number
@enum ImageType Color=1 Depth=2 Normal=3
function CDNImage(RGB::Array{T1, 3}, D::Array{T2, 3}, N::Array{T3, 3}) where T1 where T2 where T3
    if size(RGB,3) == 3
        RGB = permutedims(RGB./255.0, [3 1 2])
    end
    if size(D, 3) == 3
        D = permutedims(D, [3 1 2])
    end
    if size(N, 3) == 3
        N = permutedims(N, [3 1 2])
    end
    return cat(RGB, D,  N, dims=1) 
end
function number_of_pixels(img::CDNImage)
    return size(img,2)*size(img,3)
end
function colors(img::CDNImage)
    return img[1:3, :, :]
end
function distances(img::CDNImage)
    return img[4:6, :, :]
end
function z(img::CDNImage)
    return img[6, :, :]
end
function normals(img::CDNImage)
    return img[7:9, :, :]
end
struct LabelledCDNImage
    image::CDNImage
    labels::Matrix
end



