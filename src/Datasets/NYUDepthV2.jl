using RemoteFiles
"""
http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""
struct NYUDataset
    path::String
end
function NYUDataset()
    return NYUDataset(joinpath(dirname(pathof(RGBDSegmentation)), "..", "datasets", "NYU"))
end
import Base.length
function length(s::NYUDataset)
    return 1449
end
function raw_path(ds::NYUDataset)
    return joinpath(ds.path,"raw")
end
function raw_file(ds::NYUDataset)
    return joinpath(raw_path(ds), "nyu_depth_v2_labeled.mat")
end
function transformed_path(ds::NYUDataset)
    return joinpath(ds.path, "transformed")
end
function transformed_file_name(ds::NYUDataset, j::Integer)
    return "image_nyu_v2_$(j).fits"
end
function transformed_file_path(ds::NYUDataset, j::Integer)
    return joinpath(
        transformed_path(ds),
        transformed_file_name(ds, j)
    )
end   
function preprocess(ds::NYUDataset, n::Integer=0)
    c_params = camera_params(ds)
    file = matopen(raw_file(ds))
    RGB = read(file,"images")
    depths = read(file,"depths")
    labels = read(file,"labels")
    if n == 0
        n = size(labels,3)
    else
        n = min(n, size(labels, 3))
    end
    mkpath(transformed_path(ds))
    for j=1:n
        @info "Preprocessing image $j of $n"
        t_file_path = transformed_file_path(ds, j)
        if !isfile(t_file_path)
            f = FITS(t_file_path, "w");
            write(f, RGB[:,:,:,j])
            write(f, depth_plane2depth_world(depths[:,:,j],
                                             c_params["c_d"],
                                             c_params["f_d"]))
            write(f, labels[:,:,j])
            close(f)
        end
    end
end
function download(ds::NYUDataset)
    file_path = raw_file(ds)
    if !isfile(file_path)
        @info("Downloading file in $file_path")
        @RemoteFile(dataset_raw_file,
                    "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
                    dir=raw_path(ds))
        download(dataset_raw_file)

    else
        @info("Raw file already present")
    end
    preprocess(ds)
end
"""
```julia
function get_image(ds::NYUDataset, i::Integer; compute_normals=true)
```
"""
function get_image(ds::NYUDataset, i::Integer; compute_normals=true)
    image_path = joinpath(transformed_path(ds), string("image_nyu_v2_",i,".fits"))
    f = FITS(image_path, "r+")
    N = nothing
    D = read(f[2])
    if (length(f)==3) || compute_normals
        (N1, N, N2) = compute_local_planes(D,window=3,relDepthThresh=0.01)
    else
#        N = read(f[4])
    end
    margin = 15
    RGB = read(f[1]);
    RGB = RGB[margin:end-margin, margin:end-margin,:] ;
    D = D[margin:end-margin, margin:end-margin, :];
    N = N[margin:end-margin, margin:end-margin,:];
    S = read(f[3])
    close(f)
    return LabelledCDNImage(CDNImage(RGB,D,N),S)
end

function camera_params(ds::NYUDataset)
    params = Dict(
        # The maximum depth used, in meters.
        "maxDepth" => 10,
        # RGB Intrinsic Parameters
        "f_rgb" => [5.1885790117450188e+02 5.1946961112127485e+02],
        "c_rgb" => [3.2558244941119034e+02 2.5373616633400465e+02],
        # RGB Distortion Parameters
        "k_rgb" => [ 2.0796615318809061e-01,
                     -5.8613825163911781e-01,
                     4.9856986684705107e-01],
        "p_rgb" => [7.2231363135888329e-04  1.0479627195765181e-03],
        # Depth Intrinsic Parameters
        "f_d" => [5.8262448167737955e+02 5.8269103270988637e+02],
        "c_d" => [3.1304475870804731e+02 2.3844389626620386e+02],
        
        "k_d" => [-9.9897236553084481e-02,
                  3.9065324602765344e-01,
                  -5.1031725053400578e-01],
        "p_d" => [1.9290592870229277e-03 -1.9422022475975055e-03],
            # 3D Translation
        "t" => [2.5031875059141302e-02,
                -2.9342312935846411e-04,
                6.6238747008330102e-04],
        # Parameters for making depth absolute.
        "depthParam1" => 351.3,
        "depthParam2" => 1092.5

    )
# Rotation
    R = -[ 9.9997798940829263e-01, 5.0518419386157446e-03,
           4.3011152014118693e-03, -5.0359919480810989e-03,
           9.9998051861143999e-01, -3.6879781309514218e-03,
           -4.3196624923060242e-03, 3.6662365748484798e-03,
           9.9998394948385538e-01 ];

    R = reshape(R, (3, 3));
    R = inv(R');
    params["R"]= R
    return params

end
