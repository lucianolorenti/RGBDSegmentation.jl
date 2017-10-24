using PCL
using StatsBase
export get_normal_image,
       get_normal_SVD,
       rgb2lab,
	   rgb_plane2rgb_world,
       compute_local_planes

"""
 Calibrated using the RGBDemo Calibration tool:
   http://labs.manctl.com/rgbdemo/

"""
module CameraParams

# The maximum depth used, in meters.
maxDepth = 10;

# RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02;
fy_rgb = 5.1946961112127485e+02;
cx_rgb = 3.2558244941119034e+02;
cy_rgb = 2.5373616633400465e+02;

# RGB Distortion Parameters
k1_rgb =  2.0796615318809061e-01;
k2_rgb = -5.8613825163911781e-01;
p1_rgb = 7.2231363135888329e-04;
p2_rgb = 1.0479627195765181e-03;
k3_rgb = 4.9856986684705107e-01;

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02;
fy_d = 5.8269103270988637e+02;
cx_d = 3.1304475870804731e+02;
cy_d = 2.3844389626620386e+02;

# RGB Distortion Parameters
k1_d = -9.9897236553084481e-02;
k2_d = 3.9065324602765344e-01;
p1_d = 1.9290592870229277e-03;
p2_d = -1.9422022475975055e-03;
k3_d = -5.1031725053400578e-01;

# Rotation
R = -[ 9.9997798940829263e-01, 5.0518419386157446e-03,
   4.3011152014118693e-03, -5.0359919480810989e-03,
   9.9998051861143999e-01, -3.6879781309514218e-03,
   -4.3196624923060242e-03, 3.6662365748484798e-03,
   9.9998394948385538e-01 ];

R = reshape(R, 3, 3);
R = inv(R');

# 3D Translation
t_x = 2.5031875059141302e-02;
t_z = -2.9342312935846411e-04;
t_y = 6.6238747008330102e-04;

# Parameters for making depth absolute.
depthParam1 = 351.3;
depthParam2 = 1092.5;

end
function  get_projection_mask()
	mask = fill(false,480, 640);
	mask[45:471, 41:601] = true;
	sz = [427 561]
	return mask,sz
end
function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where T
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end
function rgb_plane2rgb_world(imgDepth)
  (H, W) = size(imgDepth);

  (xx, yy) = meshgrid(1:W, 1:H);

  XYZ = zeros(Float32,H,W,3)
  XYZ[:,:,1]  = (xx - CameraParams.cx_rgb) .* imgDepth / CameraParams.fx_rgb;
  XYZ[:,:,2]  = (yy - CameraParams.cy_rgb) .* imgDepth / CameraParams.fy_rgb;
  XYZ[:,:,3]  = imgDepth;

  return XYZ
end
"""
```julia
function    compute_local_planes(X, Y, Z, params)
```

 Computes local surface normal information. Note that in this file, the Y
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
function    compute_local_planes(XYZ;window::Integer=3, relDepthThresh::Float64 = 0.8)
    ( a , sz) = get_projection_mask()
    #H = sz[1];
    #W = sz[2];
	H = size(XYZ,1)
    W = size(XYZ,2)
    N = H * W;

    imgPlanes = zeros(Float32,H,W,4)
    imgConfs  = zeros(Float32,H,W)
	imgNormals = zeros(Float32,H,W,3)
    for pos in Base.CartesianRange((H,W))
        r = pos[1]
        c = pos[2]
        local wc = max(c-window,1):min(c+window,W)
        local wr = max(r-window,1):min(r+window,H)
        local idxs = collect(Base.Iterators.filter.((n)->abs(XYZ[n[1],n[2],3] - XYZ[r,c,3]) < XYZ[r,c,3] * relDepthThresh, CartesianRange((wr,wc))))

        if length(idxs) < 3
	        continue;
		end
		local A = ones(4,length(idxs))
		local j = 1
		for index in idxs
		    A[1:3,j] = XYZ[index[1],index[2],:]
			j=j+1
		end
    	(l, eigv) = eig(A*A');
		v = eigv[:,1]
		len = norm(v[1:3])
		imgPlanes[r,c,:]= v ./ (len+eps())
		imgNormals[r,c,:]=imgPlanes[r,c,1:3]
		imgConfs[r,c] = 1 - sqrt(max(l[1],0) / l[2]);

    end
    for pos in Base.CartesianRange((H,W))
        local r =pos[1]
        local c =pos[2]
        if norm(imgNormals[r,c,:] - [0.0;0;0]) < 0.000001
            local wc = max(c-window,1):min(c+window,W)
	        local wr = max(r-window,1):min(r+window,H)
            local  v = zeros(3)
            for pos_w in Base.CartesianRange((wr,wc))
                v=v+imgNormals[pos_w[1],pos_w[2],:]
            end
            v=v/norm(v)
            imgNormals[r,c,:]=v
        end
    end
	return (imgPlanes, imgNormals, imgConfs)
end

function clean_normal_image(img)
    (z,nr,nc) = size(img)
    for c = 1:nc
        for r =1:nr
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
    end

    return img
end
"""
```julia
get_normal_image(R, ttype;  normalSmoothingSize=3, maxDepthChangeFactor=0.2)
```
Dada una imagen de rango `R`, el tipo de algoritmo utilizado para obtener la imagen con vectores normales `ttype` y un conjunto de parámetros almacenados la función devuelve una por cada pixel una aproximación al vector normal en ese punto.

Los posibles tipos de `ttype` son `Covariance_Matrix`, `Average_3D_Gradient`, `Average_Depth_Change`
"""
function get_normal_image(R, ttype::PCL.IntegralImageNormalEstimationMethod ;  normalSmoothingSize=3.0, maxDepthChangeFactor=0.2)
    local rectSize = round(Integer, 3)
    local cfg  = IntegralImageNormalEstimation(ttype,
                                         rectSize=[rectSize; rectSize],
                                         depthDependentSmoothing=true,
                                         normalSmoothingSize=normalSmoothingSize,
                                         maxDepthChangeFactor=maxDepthChangeFactor)
    local N =  PCL.compute_normals(cfg, R)
    fill_normal!(N)
    return N
end
"""
```
function fill_normal(N::Array{T,3}; w::Integer=2) where T
```
Fill the invalid values of a normal image with the mean direction of its vicinity of radius `w`
"""
function fill_normal!(N::Array{T,3}; w::Integer=2) where T
    local nanN = isnan.(N)
    nanN = nanN[:,:,1] .| nanN[:,:,2] .| nanN[:,:,3]
    invalid_r, invalid_c = ind2sub(size(nanN),find(nanN))
    for j=1:length(invalid_r)
        local r=invalid_r[j]
        local c=invalid_c[j]
        local wr = max(r-w,1):min(r+w,size(nanN,1))
        local wc = max(c-w,1):min(c+w,size(nanN,2))
        local n  = zeros(3)
        for npos in CartesianRange((wr,wc))
            if all(.!(isnan.(N[npos[1],npos[2],:])))
                n  = n + N[npos[1],npos[2],:]
            end
        end
        n = n /norm(n)
        N[r,c,:] = vec(n)
    end
end
function get_normal_image(R,ttype::PCL.IntegralImageNormalEstimationMethod , params::Dict{<:Any,<:Any})
    normalSmoothingSize = params["normalSmoothingSize"]
    maxDepthChangeFactor = params["maxDepthChangeFactor"]
    return get_normal_image(R,ttype,  normalSmoothingSize=normalSmoothingSize, maxDepthChangeFactor=maxDepthChangeFactor)
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

    local B = float(RGB_image[:,:,3]);
    local G = float(RGB_image[:,:,2]);
    local R = float(RGB_image[:,:,1]);
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
    local RGB = zeros(3)
	for num=1:3
        local value = convert(Float64,input_color[num]) / 255.0
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
		local value = XYZ[num]
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
