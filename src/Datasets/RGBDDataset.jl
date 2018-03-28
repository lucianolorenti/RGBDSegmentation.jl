export NYUDataset,
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
    normals
import Base.download
using FITSIO
using MAT
using ..RGBDSegmentation
using ColorTypes
using StaticArrays
using Pkg
import Base: show, +,-,*,/,getindex, setindex!,zeros, zero
import ColorTypes: comp1, comp2, comp3
import Images: accum

const CDNImage{T} = Array{T, 3} where T<:Number
@enum ImageType Color=1 Depth=2 Normal=3
function CDNImage(RGB::Array{T1, 3}, D::Array{T2, 3}, N::Array{T3, 3}) where T1 where T2 where T3
    RGB = permutedims(RGB./255.0, [3 1 2])
    D = permutedims(D, [3 1 2])
    N = permutedims(N, [3 1 2])
    return cat(RGB, D,  N, dims=1) 
end
include("NYUDepthV2.jl")

function rgb2lab!(img::CDNImage{T}) where T
    aimg = to_array(img)
    for R in CartesianRange(size(img))
        aimg[1:3,R[1],R[2]] = rgb2lab(img[R][1:3])
    end
end
function colors(img::CDNImage)
    return img[1:3, :, :]
end
function distances(img::CDNImage)
    return img[4:6, :, :]
end
function normals(img::CDNImage)
    return img[7:9, :, :]
end
function color_image(img::CDNImage)
    res = Matrix{RGB{Float32}}(size(img))
    for I in CartesianRange(size(img))
        res[I]=RGB((img[I][1:3])...)
    end
    return res
end
function z_image(img::CDNImage)
    res = Matrix{Float32}(size(img))
    for I in CartesianRange(size(img))
        res[I]=img[I][6]
    end
    return res
end
struct LabelledCDNImage
    image::CDNImage
    labels::Matrix
end
function resize(img::CDNImage, scale::Float64)
    dim = (round(Integer,size(img.depth,1) * scale), round(Integer,size(img.depth,2)*scale))
    RGB = imresize(RGB, (dim[1],dim[2]))
    D   = imresize(D, (dim[1],dim[2]))
    N_n = zeros(dim[1],dim[2],3)
    for i=1:3
        N_n[:,:,i] = imresize(N[:,:,i], (dim[1],dim[2]))
    end
    return CDNImage(RGB,D,N_n)
end

