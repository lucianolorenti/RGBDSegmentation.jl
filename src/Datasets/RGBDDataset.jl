export RGBDImage,
    NYUDataset,
    get_image,
    RGBDN,
    RGBDNImage,
    LabelledRGBDNImage
import Base.download
using FITSIO
using MAT
using PCL
using ..RGBDSegmentation
using ColorTypes
using StaticArrays
immutable RGBDN{T} <: Color{T, 9}
    data::SVector{9,T}
end
import Base: -, start, done, next, zero, /, + 

function RGBDN{T}(a::RGBDN{T}) where T 
    return RGBDN{T}(a.data)
end
start(a::RGBDN) =  start(a.data)
done(a::RGBDN, state) = done(a.data, state)
next(a::RGBDN, state) = next(a.data, state)
zero(::Type{RGBDN{T}}) where T<:Number = RGBDN{T}(SVector{9,T}(zeros(9)))
/(d::RGBDN{T1}, a::T2) where T1<:Number where T2<:Number = RGBDN{T1}(d.data / a)
-(d1::RGBDN{T}, d2::T2) where T where T2<:Number =  RGBDN{T}(d1.data-d2)
-(d1::RGBDN{T}, d2::RGBDN{T}) where T = RGBDN{T}( d1.data - d2.data)
+(d1::RGBDN{T}, d2::T2) where T where T2<:Number =  RGBDN{T}(d1.data + d2)
+(d1::RGBDN{T}, d2::RGBDN{T}) where T = RGBDN{T}( d1.data + d2.data)
import Base.show
import Base.getindex
function getindex(c::RGBDN, i::Integer)
    return c.data[i]
end
function show(io::IO, c::RGBDN)
    (r,g,b,x,y,z,nx,ny,nz) = (c.data...)
    print(io, "rgb: [$r, $g, $b] | xyz: [$x, $y, $z] | n: [$nx, $ny, $nz]")
end
const RGBDNImage{T} = Matrix{RGBDN{T}} where T<:Number
const RGBDImage{T}  = Matrix{T} where T<:Number
import Base.convert
function convert(::Type{RGBDNImage}, m::Array{T,3} ) where T<:Number
    (nr,nc,_) = size(m)
    local img = RGBDNImage{T}(nr,nc)
    @inbounds for r=1:nr
        @inbounds for c=1:nc
            @inbounds img[r,c] = RGBDN(SVector{9,T}(m[r,c,:]))
        end
    end
    return img
end
immutable LabelledRGBDNImage
    image::RGBDNImage
    labels::Matrix
end
function LabelledRGBDNImage(RGB::Array, D::Array, N::Array, labels::Matrix)
    local  b= RGBDNImage(cat(3,RGB,D,N))
    println(typeof(b))
    return LabelledRGBDNImage(RGBDNImage(cat(3,RGB,D,N)), labels)
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
    RGB = RGB[margin:end-margin, margin:end-margin,:];
    D   = D[margin:end-margin, margin:end-margin, :];
    N   = N[margin:end-margin, margin:end-margin,:];
    S   = read(f[3])
    close(f)
    return LabelledRGBDNImage(RGB,D,N,S)
end
