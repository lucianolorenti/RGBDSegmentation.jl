using StatsBase
using LinearAlgebra
using Images
using ImageSegmentation
export get_normal_image,
    get_normal_SVD,
    rgb2lab,
    rgb_plane2rgb_world,
    compute_local_planes,
    resize

function  get_projection_mask()
    mask = fill(false, 480, 640);
    mask[45:471, 41:601] .= true;
    sz = [427 561]
    return mask,sz
end
"""
 Projects the given depth image to world coordinates. Note that this 3D
 coordinate space is defined by a horizontal plane made from the X and Z
 axes and the Y axis points up.

 Args:
   imgDepthAbs - 480x640 depth image whose values indicate depth in
                meters.

 Returns:
   points3d - Nx3 matrix of 3D world points (X,Y,Z).

"""
function depth_plane2depth_world(imgDepthAbs, c_d, f_d)
    (H, W) = size(imgDepthAbs);
    X = (collect(1:W)' .- c_d[1]) .* imgDepthAbs / f_d[1];
    Y = (collect(1:H) .- c_d[2]) .* imgDepthAbs / f_d[2];
    Z = imgDepthAbs
    return cat(X,Y,Z, dims=3)
end
"""
```julia
function compute_local_planes(X, Y, Z, params)
```

 Computes surface normal information. Note that in this file, the Y
 coordinate points up, consistent with the image coordinate frame.

 Args:
   X - Nx1 column vector of 3D point cloud X-coordinates
   Y - Nx1 column vector of 3D point cloud Y-coordinates
   Z - Nx1 column vector of 3D point cloud Z-coordinates.
   params

 Returns:
   - `imgPlanes`. An 'image' of the plane parameters for each pixel.
   - `imgNormals`. HxWx3 matrix of surface normals at each pixel.
   - `imgConfs`. HxW image of confidences.
"""
function compute_local_planes(XYZ;window::Integer=3, relDepthThresh::Float64 = 0.8)
    ( a , sz) = get_projection_mask()
    #H = sz[1];
    #W = sz[2];
	H = size(XYZ,1)
    W = size(XYZ,2)
    N = H * W;

    imgPlanes = zeros(Float32,H,W,4)
    imgConfs  = zeros(Float32,H,W)
	imgNormals = zeros(Float32,H,W,3)
    for r=1:H, c=1:W
        wc = max(c-window,1):min(c+window,W)
        wr = max(r-window,1):min(r+window,H)
        idxs = [(rwr, cwc)  for rwr in wr, cwc in wc
                 if abs(XYZ[rwr,cwc,3] - XYZ[r, c, 3]) < XYZ[r,c,3] * relDepthThresh ]
        if length(idxs) < 3
	    continue;
	end
	A = ones(4,length(idxs))
	j = 1
	for (rwr, rwc) in idxs
	    A[1:3,j] = XYZ[rwr,rwc,:]
	    j=j+1
	end
    	(l, eigv) = eigen(A*A');
	v = eigv[:,1]
	len = norm(v[1:3])
	imgPlanes[r,c,:]= v ./ (len+eps())
	imgNormals[r,c,:]=imgPlanes[r,c,1:3]
	imgConfs[r,c] = 1 - sqrt(max(l[1],0) / l[2]);
    end
    for r=1:H, c=1:W
        if norm(imgNormals[r,c,:] - [0.0;0;0]) < 0.000001
            wc = max(c-window,1):min(c+window,W)
	    wr = max(r-window,1):min(r+window,H)
            v = zeros(3)
            for pos_wr in wr, pos_wc in wc
                v=v+imgNormals[pos_wr,pos_wc,:]
            end
            v=v/norm(v)
            imgNormals[r,c,:]=v
        end
    end
    return (imgPlanes, imgNormals, imgConfs)
end

function clean_normal_image(img)
    (z,nr,nc) = size(img)
    for c = 1:nc, c=1:nr
        if  any(.!(isfinite.(img[:,r,c])))
            wr = max(r-3,1):min(r+3,nr)
            wc = max(c-3,1):min(c+3,nc)
            imgw = img[:,wr,wc]
            imgw_b = sum(!isfinite.(imgw),1)
            cc = reshape(imgw_b,length(wr),length(wc))
            bb =     find(cc.== 0)
            (fs,cs) = ind2sub(size(cc),bb)
            v = zeros(3)
            for (i,j) in zip(fs,cs)
                v=v+imgw[:,i,j]
            end
            v=v/length(fs)
            v=v/norm(v)
            img[:,r,c]=v
        end
    end
    return img
end
"""
```
function fill_normal(N::Array{T,3}; w::Integer=2) where T
```
Fill the invalid values of a normal image with the mean direction of its vicinity of radius `w`
"""
function fill_normal!(N::Array{T,3}; w::Integer=2) where T
    nanN = isnan.(N)
    nanN = nanN[:,:,1] .| nanN[:,:,2] .| nanN[:,:,3]
    invalid_r, invalid_c = ind2sub(size(nanN),find(nanN))
    for j=1:length(invalid_r)
        r=invalid_r[j]
        c=invalid_c[j]
        wr = max(r-w,1):min(r+w,size(nanN,1))
        wc = max(c-w,1):min(c+w,size(nanN,2))
        n  = zeros(3)
        for nposr=1:wr, nposc=1:wc
            if all(.!(isnan.(N[nposr,nposc,:])))
                n  = n + N[nposr,nposc,:]
            end
        end
        n = n /norm(n)
        N[r,c,:] = vec(n)
    end
end
function get_normal_SVD(R, w::Integer, th::Float64)
    nr = size(R,1)
    nc = size(R,2)
    N  = zeros(Float32,nr,nc,3)
    for c = 1:nc
        for r=1:nr
            wc = max(c-w,1):min(c+w,nc)
            wr = max(r-w,1):min(r+w,nr)
            indexes    = Base.CartesianRange((wr,wc))
            x = Float32[]
            y = Float32[]
            z = Float32[]
            for V in indexes
                (wwr, wwc)   = (V[1], V[2])
                if abs(R[r,c,3]-R[wwr,wwc,3]) <= th
                    push!(x,R[wwr,wwc,1])
                    push!(y,R[wwr,wwc,2])
                    push!(z,R[wwr,wwc,3])
                end
            end
            if (length(x) > 3)
                push!(x,R[r,c,1])
                push!(y,R[r,c,2])
                push!(z,R[r,c,3])

                M = hcat(x,y,z)'
                M = M .- mean(M,2)

                (U, A2, V) = svd(M)
                N[r,c,:]= U[:,3] / norm(U[:,3])
            end
        end
    end
    return N
end
function get_normal_image(R, ttype::String; d::Integer = 3)
    w  = params["d"]
    if ttype=="SVD"
    end
end

#="""

"""
function rgb2lab(RGB_image::Array{T,3}) where T
	(nr, nc, c) = size(RGB_image)
	res         = zeros(Float16, (nr,nc,3))
	for i in CartesianRange((nr,nc))
		res[i[1],i[2],:] = rgb2lab(RGB_image[i[1],i[2],:])
	end
	return res
end=#
"""
```
function rgb2lab(RGB_image::Array{T,3}) where T
```
% RGB2Lab takes matrices corresponding to Red, Green, and Blue, and
% transforms them into CIELab.  This transform is based on ITU-R
% Recommendation  BT.709 using the D65 white point reference.
% The error in transforming RGB -> Lab -> RGB is approximately
% 10^-5.  RGB values can be either between 0 and 1 or between 0 and 255.
% By Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
% Updated for MATLAB 5 28 January 1998.
"""
function  rgb2lab(RGB_image::Array{ElType,3}) where ElType

    B = float(RGB_image[:,:,3]);
    G = float(RGB_image[:,:,2]);
    R = float(RGB_image[:,:,1]);
    if ((maximum(R) > 1.0) | (maximum(G) > 1.0) | (maximum(B) > 1.0))
        R = R./255.0;
        G = G./255.0;
        B = B./255.0;
    end
    (M, N) = size(R);
    s = M*N;
    #Set a threshold
    T = 0.008856;

    RGB = hcat(reshape(R,s), reshape(G,s), reshape(B,s))';
    # RGB to XYZ
    MAT = [0.412453 0.357580 0.180423;
           0.212671 0.715160 0.072169;
           0.019334 0.119193 0.950227];
    XYZ = MAT * RGB;

    X = XYZ[1,:] / 0.950456;
    Y = XYZ[2,:]
    Z = XYZ[3,:] / 1.088754;

    XT = X .> T;
    YT = Y .> T;
    ZT = Z .> T;

    fX = XT .* X.^(1/3) + (.!XT) .* (7.787 .* X + 16/116);

    # Compute L
    Y3 = Y.^(1/3);
    fY = YT .* Y3 + (.!YT) .* (7.787 .* Y + 16/116);
    L  = YT .* (116 * Y3 - 16.0) + (.!YT) .* (903.3 * Y);

    fZ = ZT .* Z.^(1/3) + (.!ZT) .* (7.787 .* Z + 16/116);

    #    Compute a and b
    a =     500 * (fX - fY);
    b = 200 * (fY - fZ);

    L = reshape(L, M, N);
    a = reshape(a, M, N);
    b = reshape(b, M, N);

    return L = cat(3,L,a,b);
end
"""

"""
function rgb2lab(input_color::Vector)
    RGB = zeros(3)
    for num=1:3
        value = convert(Float64,input_color[num]) / 255.0
        if value > 0.04045
            value = ((value + 0.055) / 1.055) ^ 2.4
        else
            value = value / 12.92
		end
        RGB[num] = value * 100
    end
    XYZ = zeros(3)
    XYZ[1] = RGB[1] * 0.4124 + RGB[2] * 0.3576 + RGB[3] * 0.1805
    XYZ[2] = RGB[1] * 0.2126 + RGB[2] * 0.7152 + RGB[3] * 0.0722
    XYZ[3] = RGB[1] * 0.0193 + RGB[2] * 0.1192 + RGB[3] * 0.9505

    # Observer= 2°, Illuminant= D65
    XYZ[1] = XYZ[1] / 95.047         # ref_X =  95.047
    XYZ[2] = XYZ[2] / 100.0          # ref_Y = 100.000
    XYZ[3] = XYZ[3] / 108.883        # ref_Z = 108.883

    num = 0
    for num=1:3
	value = XYZ[num]
        if value > 0.008856
            value = value ^ (0.3333333333333333)
        else
            value = (7.787 * value) + (16 / 116.0)
	end
        XYZ[num] = value
    end
    Lab = zeros(3)

    L = (116 * XYZ[2]) - 16
    a = 500 * (XYZ[1] - XYZ[2])
    b = 200 * (XYZ[2] - XYZ[3])

    Lab[1] = L
    Lab[2] = a
    Lab[3] = b
    return Lab
end
function rgb2lab!(img::CDNImage{T}) where T
    for r=1:size(img,2), c=1:size(img,3)
        img[1:3,r,c] = rgb2lab(img[1:3,r,c])
    end
end
function resize(labels::Matrix{T}, scale::Float64) where T<:Integer
    dim = (round(Integer, size(labels, 1) * scale),
           round(Integer, size(labels, 2) * scale))
    return round.(Integer, imresize(labels, dim))
end
function resize(img::CDNImage, scale::Float64)
    D = distances(img)
    dim = (round(Integer, size(D, 2) * scale),
           round(Integer, size(D, 3) * scale))
    RGB = Array(channelview(imresize(colors(img), dim)))
    D = imresize(distances(img), (3, dim...))
    N = imresize(normals(img), (3, dim...))
    return CDNImage(RGB,D,N)
end
function resize(img::LabelledCDNImage, scale::Float64)
    resized_img = resize(img.image, scale)
    resized_labels = resize(labels_map(img.ground_truth), scale)
    return LabelledCDNImage(
        resized_img,
        resized_labels)
end
 
