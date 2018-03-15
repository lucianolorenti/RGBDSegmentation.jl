export RGBDImage,
    NYUDataset,
    get_image,
    RGBDN,
    RGBDNImage,
    RGBDNColorant,
    LabelledRGBDNImage,
    color_image,
    z_image,
    to_array,
    to_rgbdnimage,
    to_image,
    rgb2lab!
import Base.download
using FITSIO
using MAT
using PCL
using ..RGBDSegmentation
using ColorTypes
using StaticArrays
using TypedDelegation   
import Base: show, +,-,*,/,getindex, setindex!,zeros, zero
import ColorTypes: comp1, comp2, comp3
struct RGBDN{T} <: Colorant{T,9}
    data::SVector{9,T}
end
comp1(d::RGBDN{T}) where T = d[1]
comp2(d::RGBDN{T}) where T = d[2]
comp3(d::RGBDN{T}) where T = d[3]
function zero(::Type{RGBDN{T}}) where T
    return RGBDN{T}(zero(SVector{9,T}))
end
@delegate_onefield(RGBDN, data, [getindex])
@delegate_onefield_astype(RGBDN, data, [zeros,+, -, /,*])
@delegate_onefield_twovars_astype( RGBDN, data, [ +,-,/, *] );
import Images: accum
accum(d::Type{RGBDN{T}}) where T = RGBDN{T}
function show(io::IO, c::RGBDN)
    (r,g,b,x,y,z,nx,ny,nz) = (c.data...)
    print(io, "rgb: [$r, $g, $b] | xyz: [$x, $y, $z] | n: [$nx, $ny, $nz]")
end
const RGBDNImage{T} = Matrix{RGBDN{T}} where T<:Number
const RGBDImage{T}  = Matrix{T} where T<:Number


function rgb2lab!(img::RGBDNImage{T}) where T
    aimg = to_array(img)
    for R in CartesianRange(size(img))
        aimg[1:3,R[1],R[2]] = rgb2lab(img[R][1:3])
    end
end
function to_array(img::RGBDNImage{T}) where T
    return reinterpret(Float64, img, (9, size(img,1),size(img,2)))
end
function to_image(img::RGBDNImage{T}) where T
    return reinterpret(RGBDNColorant{T}, img, (size(img)))
end
function to_rgbdnimage(img::Array{T,3}) where T
    return reinterpret(RGBDN{T}, img, (size(img,2),size(img,3)))
end
function color_image(img::RGBDNImage)
    local res = Matrix{RGB{Float32}}(size(img))
    for I in CartesianRange(size(img))
        res[I]=RGB((img[I][1:3])...)
    end
    return res
end
function z_image(img::RGBDNImage)
    local res = Matrix{Float32}(size(img))
    for I in CartesianRange(size(img))
        res[I]=img[I][6]
    end
    return res
end
immutable LabelledRGBDNImage
    image::RGBDNImage
    labels::Matrix
end
function LabelledRGBDNImage(RGB::Array, D::Array, N::Array, labels::Matrix)
    return LabelledRGBDNImage(to_rgbdnimage(permutedims(cat(3,RGB/255.0,D,N), [3 1 2])), labels)
end

function resize(img::RGBDImage, scale::Float64)
    local dim = (round(Integer,size(img.depth,1) * scale), round(Integer,size(img.depth,2)*scale))
    RGB = imresize(RGB, (dim[1],dim[2]))
    D   = imresize(D, (dim[1],dim[2]))
    N_n = zeros(dim[1],dim[2],3)
    for i=1:3
        N_n[:,:,i] = imresize(N[:,:,i], (dim[1],dim[2]))
    end
    return RGBDImage(RGB,D,N_n)
end

    """
http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""
struct NYUDataset
    path::String
end
function NYUDataset()
    return NYUDataset("/home/luciano/datasets/")
end
import Base.length
function length(s::NYUDataset)
    return 1449
end
function download(ds::NYUDataset)
    local ds_folder = joinpath(Pkg.dir("RGBDSegmentation"),"data")
    local file_path = joinpath(ds_folder, "nyu_depth_v2_labeled.mat")
    if !isfile(file_path)
        local url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
        mkpath(ds_folder)
        download(url, file_path)
    end
    file   = matopen(file_path)
    RGB    = read(file,"images")
    depths = read(file,"depths")
    labels = read(file,"labels")
    n      = size(labels,3)
    for j=1:n
        filename = "image_nyu_v2_$(j).fits"
        f = FITS(joinpath(ds.path, filename),"w");
        write(f, RGB[:,:,:,j])
        write(f, depths[:,:,j])
        write(f, labels[:,:,j])
        close(f)
    end
end
doc"""
```julia
function get_image(ds::NYUDataset, i::Integer; compute_normals=true)
```
"""
function get_image(ds::NYUDataset, i::Integer; compute_normals=false)
    local image_path = joinpath(ds.path, string("image_nyu_v2_",i,".fits"))
    local f = FITS(image_path, "r+")
    local N = nothing

    D = rgb_plane2rgb_world(read(f[2]))
    if (length(f)==3) || compute_normals
        #(N1, N, N2) = compute_local_planes(D,window=3,relDepthThresh=0.01)
        N = get_normal_image(D,PCL.Average_3D_Gradient)
    else
        N = read(f[4])
    end
    local margin = 15
    RGB = read(f[1]);
    RGB = RGB[margin:end-margin, margin:end-margin,:] ;
    D   = D[margin:end-margin, margin:end-margin, :];
    N   = N[margin:end-margin, margin:end-margin,:];
    S   = read(f[3])
    close(f)
    return LabelledRGBDNImage(RGB,D,N,S)
end
